#!/usr/bin/env python3
"""Original JAX SGD-only experiment; preserves shared locks and budget.

This is NOT a concurrent launcher. The existing SAM process must finish/stop
before --execute can acquire the shared launcher and budget locks.
"""
import importlib.util
from pathlib import Path
import sys


def main():
    source = Path(__file__).resolve().parents[1]/'cifar10_wrn28x10_sam_sgd_50epochs_seed0'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('original_50epoch_launcher', source/'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.METHODS = ('SGD-50',)
    runner.SESSION_TREE = 'runs_sgd50'
    runner.SESSION_POINTER = 'latest_sgd50_session.txt'
    runner.main()


if __name__ == '__main__':
    main()
