"""Run independent task shards of eval.py, then validate and merge results."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from omegaconf import OmegaConf


def merge_results(paths, expected_tasks, checkpoint, test_num):
    tasks = {}
    protocol = None
    for path in paths:
        result = json.loads(Path(path).read_text())
        if result['checkpoint'] != str(Path(checkpoint).resolve()):
            raise ValueError(f'{path}: checkpoint mismatch')
        signature = (result.get('seed'), result.get('horizon'), result.get('test_num'), result.get('rng_protocol'))
        if protocol is not None and signature != protocol:
            raise ValueError(f'{path}: evaluation protocol mismatch')
        protocol = signature
        for task, item in result['tasks'].items():
            if task in tasks:
                raise ValueError(f'Duplicate task result: {task}')
            if item['episodes'] != test_num or not 0 <= item['success'] <= test_num:
                raise ValueError(f'{task}: invalid episode counts')
            expected_rate = item['success'] / test_num
            if abs(item['success_rate'] - expected_rate) > 1e-8:
                raise ValueError(f'{task}: inconsistent success rate')
            tasks[task] = item
    if set(tasks) != set(expected_tasks):
        raise ValueError('Missing or unexpected tasks; refusing to report partial success rates')
    tasks = {task: tasks[task] for task in expected_tasks}
    return dict(checkpoint=str(Path(checkpoint).resolve()), seed=protocol[0],
                horizon=protocol[1], test_num=test_num, rng_protocol=protocol[3],
                tasks=tasks,
                overall_success_rate=sum(x['success_rate'] for x in tasks.values()) / len(tasks),
                total_success=sum(x['success'] for x in tasks.values()),
                total_episodes=sum(x['episodes'] for x in tasks.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads-per-worker', type=int, default=1)
    p.add_argument('--test-num', type=int, default=50)
    p.add_argument('--horizon', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true', help='print shards and commands without starting evaluation')
    args = p.parse_args()
    if min(args.workers, args.threads_per_worker, args.test_num, args.horizon) <= 0:
        p.error('workers, threads, test-num and horizon must be positive')
    cfg = OmegaConf.load(args.config)
    tasks = args.tasks or list(cfg.dataset.task_names)
    if not tasks or len(tasks) != len(set(tasks)):
        p.error('tasks must be nonempty and unique')
    if cfg.dataset.get('online_eval_verified', True) is False:
        p.error('External dataset online evaluation is not calibrated')
    for task in tasks:
        if task not in cfg.dataset.task_names:
            p.error(f'Task not present in config: {task}')
    for path in (args.config, args.stats, args.checkpoint):
        if not Path(path).is_file():
            p.error(f'Missing file: {path}')
    if args.output.exists():
        p.error(f'Output exists, choose a new filename: {args.output}')
    workers = min(args.workers, len(tasks))
    shards = [tasks[i::workers] for i in range(workers)]
    run_dir = args.output.parent / f'{args.output.stem}_parts_{uuid.uuid4().hex[:8]}'
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    env['OMP_NUM_THREADS'] = str(args.threads_per_worker)
    env['MKL_NUM_THREADS'] = str(args.threads_per_worker)
    env['OPENBLAS_NUM_THREADS'] = str(args.threads_per_worker)
    commands = []
    results = []
    for i, shard in enumerate(shards):
        output = run_dir / f'worker_{i:02d}.json'
        command = [sys.executable, '-u', str(Path(__file__).with_name('eval.py')),
                   '--config', args.config, '--stats', args.stats,
                   '--checkpoint', args.checkpoint, '--output', str(output),
                   '--test-num', str(args.test_num), '--horizon', str(args.horizon),
                   '--seed', str(args.seed), '--device', args.device, '--tasks', *shard]
        commands.append(command)
        results.append(output)
        print(f'Worker {i}: {len(shard)} tasks; log={run_dir / f"worker_{i:02d}.log"}', flush=True)
        if args.dry_run:
            print(shlex.join(command))
    if args.dry_run:
        return
    run_dir.mkdir(parents=True)
    (run_dir / 'manifest.json').write_text(json.dumps(dict(commands=commands, task_shards=shards,
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'), threads_per_worker=args.threads_per_worker), indent=2) + '\n')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for i, command in enumerate(commands):
            log = (run_dir / f'worker_{i:02d}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
        finished = set()
        while len(finished) < workers:
            for i, process in enumerate(processes):
                status = process.poll()
                if status is None or i in finished:
                    continue
                if status != 0:
                    raise RuntimeError(f'Worker {i} failed (exit {status}); see {run_dir / f"worker_{i:02d}.log"}')
                finished.add(i)
                print(f'Worker {i} finished ({len(finished)}/{workers})', flush=True)
            time.sleep(0.5)
        merged = merge_results(results, tasks, args.checkpoint, args.test_num)
        merged.update(workers=workers, wall_seconds=time.monotonic() - started, parts_dir=str(run_dir))
        args.output.write_text(json.dumps(merged, indent=2) + '\n')
        print(f'Saved {args.output}: {merged["total_success"]}/{merged["total_episodes"]}', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
