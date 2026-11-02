"""Unfrozen scene evaluation with the legacy parallel console/log layout."""
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
from eval_parallel import merge_results
from reuse_options import add_reuse_arguments, reuse_cli, validate_reuse_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'stats', 'checkpoint'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=50)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    add_reuse_arguments(parser)
    args = parser.parse_args()
    validate_reuse_arguments(parser, args)
    if min(args.workers, args.threads_per_worker, args.test_num, args.horizon) <= 0:
        parser.error('workers, threads, test-num and horizon must be positive')
    if args.seed < 0 or args.seed + args.test_num - 1 >= 2**32:
        parser.error('episode seeds must be in [0, 2**32)')
    cfg = OmegaConf.load(args.config)
    tasks = args.tasks or list(cfg.dataset.task_names)
    if not tasks or len(tasks) != len(set(tasks)):
        parser.error('tasks must be nonempty and unique')
    if cfg.dataset.get('online_eval_verified', True) is False:
        parser.error('External dataset online evaluation is not calibrated')
    for task in tasks:
        if task not in cfg.dataset.task_names:
            parser.error(f'Task not present in config: {task}')
    for path in (args.config, args.stats, args.checkpoint):
        if not Path(path).is_file():
            parser.error(f'Missing file: {path}')
    if args.output.exists():
        parser.error(f'Output exists, choose a new filename: {args.output}')
    workers = min(args.workers, len(tasks))
    shards = [tasks[i::workers] for i in range(workers)]
    run_dir = args.output.parent / f'{args.output.stem}_parts_{uuid.uuid4().hex[:8]}'
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(args.threads_per_worker)
    commands, results = [], []
    for i, shard in enumerate(shards):
        output = run_dir / f'worker_{i:02d}.json'
        command = [sys.executable, '-u', str(Path(__file__).with_name('eval_adapolicy.py')),
                   '--config', args.config, '--stats', args.stats,
                   '--checkpoint', args.checkpoint, '--output', str(output),
                   '--test-num', str(args.test_num), '--horizon', str(args.horizon),
                   '--seed', str(args.seed), '--device', args.device, '--tasks', *shard]
        command += reuse_cli(args)
        commands.append(command)
        results.append(output)
        print(f'Worker {i}: {len(shard)} tasks; log={run_dir / f"worker_{i:02d}.log"}', flush=True)
        if args.dry_run:
            print(shlex.join(command))
    if args.dry_run:
        return
    run_dir.mkdir(parents=True)
    (run_dir / 'manifest.json').write_text(json.dumps(dict(
        commands=commands, task_shards=shards, evaluator='eval_adapolicy.py',
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
        threads_per_worker=args.threads_per_worker), indent=2) + '\n')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for i, command in enumerate(commands):
            stream = (run_dir / f'worker_{i:02d}.log').open('w')
            logs.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env))
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
        print(f'Saved {args.output}: {merged["total_success"]}/{merged["total_episodes"]} '
              f'({merged["overall_success_rate"]:.2%})', flush=True)
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
        for stream in logs:
            stream.close()


if __name__ == '__main__':
    main()
