"""Portable entry points for baseline reproduction and robustness experiments."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
KINDS = {'node-missing': ['node'], 'temporal-gap': ['node_gap'],
         'gaussian-noise': ['noise'], 'robustness': ['node', 'node_gap', 'noise'],
         'clean': ['clean']}
MODULES = ['numpy', 'torch', 'pandas', 'scipy', 'sklearn', 'torch_geometric',
           'pywt', 'easytorch', 'easydict', 'packaging', 'setproctitle', 'tqdm',
           'tensorboard', 'matplotlib']


def dependencies(install):
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if missing and install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)
        missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing packages: {missing}. Run again with --install-deps.')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The original repository's --dataset/--data-root command remains usable.
    if argv and argv[0].startswith('-') and argv[0] not in ('-h', '--help'):
        argv.insert(0, 'reproduce')
    parser = argparse.ArgumentParser(description='STDN baseline and STDN/HyperD robustness experiments; STWave uses plot_stwave_12steps.py')
    parser.add_argument('experiment', choices=['reproduce', *KINDS, 'analyze', 'check'])
    parser.add_argument('--datasets', '--dataset', nargs='+', default=DATASETS,
                        choices=DATASETS + ['all'])
    parser.add_argument('--models', nargs='+', choices=['STDN', 'HyperD'], default=['STDN', 'HyperD'])
    parser.add_argument('--rates', nargs='+', type=float, choices=[.1, .2, .3], default=[.1, .2, .3])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/flow')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/graphs')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--tag', help='Output group; defaults to the experiment name')
    parser.add_argument('--action', choices=['run', 'plan', 'prepare', 'summarize'], default='run')
    parser.add_argument('--plan', action='store_true', help='List combinations only; do not train')
    parser.add_argument('--smoke', action='store_true', help='Four windows per split and one epoch, execution checks only')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--corruption-seed', type=int, default=2024)
    parser.add_argument('--training-seed', '--seed', type=int)
    parser.add_argument('--include-clean', action='store_true')
    parser.add_argument('--install-deps', action='store_true')
    parser.add_argument('--batch-size', type=int, help='Only for the original STDN reproduce command')
    args = parser.parse_args(argv)
    if 'all' in args.datasets:
        if len(args.datasets) != 1:
            parser.error('Use --datasets all on its own')
        args.datasets = DATASETS
    args.data_root = args.data_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.tag is not None and (Path(args.tag).name != args.tag or args.tag in ('.', '..')):
        parser.error('--tag must be a single folder name')
    if args.experiment == 'check':
        dependencies(args.install_deps)
        subprocess.run([sys.executable, '-m', 'unittest', 'robustness.test_corruptions', '-v'], cwd=ROOT, check=True)
        return
    if args.experiment == 'analyze':
        dependencies(args.install_deps)
        from robustness.analyze import analyze
        search = args.output_root / 'robustness'
        if args.tag:
            search /= args.tag
        analyze(search, args.output_root / 'analysis' / (args.tag or 'all'), args.smoke)
        return
    if args.experiment == 'reproduce':
        if args.plan or args.action != 'run':
            print('Original STDN reproduction:', ', '.join(args.datasets), '(no training started)')
            return
        dependencies(args.install_deps)
        for dataset in (['all'] if args.datasets == DATASETS else args.datasets):
            command = [sys.executable, str(ROOT / 'models/STDN/run.py'), '--dataset', dataset,
                       '--data-root', str(args.data_root), '--raw-root', str(args.raw_root),
                       '--output-root', str(args.output_root / 'baseline' / (args.tag or 'stdn')),
                       '--cache-root', str(ROOT / '.cache/stdn'), '--device', args.device]
            if args.smoke:
                command.append('--smoke')
            if args.training_seed is not None:
                command += ['--seed', str(args.training_seed)]
            if args.batch_size is not None:
                command += ['--batch-size', str(args.batch_size)]
            subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.batch_size is not None:
        parser.error('--batch-size applies only to reproduce; robustness keeps each model config')
    action = 'plan' if args.plan else args.action
    if action not in ('plan',):
        dependencies(args.install_deps)
    device = args.device
    if device == 'auto' and action == 'run':
        import torch
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() else 'cpu'
    elif device == 'auto':
        device = 'cuda'
    tag = args.tag or args.experiment.replace('-', '_')
    output = args.output_root / 'robustness'
    command = [sys.executable, str(ROOT / 'robustness/run.py'), action,
               '--models', *args.models, '--datasets', *args.datasets,
               '--kinds', *KINDS[args.experiment], '--rates', *[str(r) for r in args.rates],
               '--data-root', str(args.data_root), '--raw-root', str(args.raw_root),
               '--output-root', str(output), '--tag', tag, '--device', device,
               '--corruption-seed', str(args.corruption_seed)]
    if args.training_seed is not None:
        command += ['--training-seed', str(args.training_seed)]
    if args.smoke:
        command.append('--smoke')
    if args.include_clean:
        command.append('--include-clean')
    subprocess.run(command, cwd=ROOT, check=True)
    if action in ('run', 'summarize'):
        from robustness.analyze import analyze
        analyze(output / tag, args.output_root / 'analysis' / tag, args.smoke)


if __name__ == '__main__':
    main()
