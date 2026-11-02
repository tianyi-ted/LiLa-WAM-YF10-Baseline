"""Shared CLI for serial and sharded inference ablations (defaults unchanged)."""
import math


def add_reuse_arguments(parser):
    parser.add_argument('--action-reuse', action='store_true',
                        help='reuse the unexecuted normalized action tail as a noisy source')
    parser.add_argument('--reuse-sigma', type=float, default=1.0)
    parser.add_argument('--reuse-strength', type=float, default=1.0)
    parser.add_argument('--num-inference-steps', type=int, default=None,
                        help='override Euler NFE without modifying the model YAML')
    parser.add_argument('--policy-seed', type=int, default=None,
                        help='optional independent per-task action-noise RNG')
    parser.add_argument('--profile-generation', action='store_true',
                        help='synchronize CUDA to measure chunk generation; adds timing overhead')


def validate_reuse_arguments(parser, args):
    for name in ('reuse_sigma', 'reuse_strength'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'{name} must be finite and nonnegative')
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error('num-inference-steps must be positive')
    if args.policy_seed is not None and args.policy_seed < 0:
        parser.error('policy-seed must be nonnegative')


def reuse_cli(args):
    flags = ['--reuse-sigma', str(args.reuse_sigma),
             '--reuse-strength', str(args.reuse_strength)]
    if args.action_reuse:
        flags.append('--action-reuse')
    if args.num_inference_steps is not None:
        flags += ['--num-inference-steps', str(args.num_inference_steps)]
    if args.policy_seed is not None:
        flags += ['--policy-seed', str(args.policy_seed)]
    if args.profile_generation:
        flags.append('--profile-generation')
    return flags
