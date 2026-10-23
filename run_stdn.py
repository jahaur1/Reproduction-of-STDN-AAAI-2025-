"""STDN-only entry: default four-dataset reproduction, optional robustness mode.

Examples:
    python run_stdn.py
    python run_stdn.py --datasets PEMS08
    python run_stdn.py gaussian-noise --datasets PEMS08 --rates 0.1
"""
import sys
from launcher import main as launch

MODES = ('reproduce', 'node-missing', 'temporal-gap', 'gaussian-noise', 'robustness', 'clean')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--models' in argv:
        raise SystemExit('run_stdn.py runs STDN only; use run.py to choose other models.')
    mode = argv.pop(0) if argv and argv[0] in MODES else 'reproduce'
    launch([mode, '--models', 'STDN', *argv])


if __name__ == '__main__':
    main()
