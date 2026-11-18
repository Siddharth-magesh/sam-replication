#!/usr/bin/env python3
"""Print the final seed-0 comparison from saved records; no training or GPU needed."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='Original SESSION or a downloaded copy.')
    args = parser.parse_args()
    errors, identities, initializations = {}, [], []
    for method, reference in [('SAM-200', 2.7), ('SGD-200', 3.5)]:
        folder = args.session / 'train' / (method + '_seed_0')
        status = json.loads((folder / 'status.json').read_text())
        rows = json.loads((folder / 'history.json').read_text())
        identity = json.loads((folder / 'manifest.json').read_text())['identity']
        identities.append((identity['config'], identity['worker_sha256']))
        initializations.append(json.loads((folder / 'initialization.json').read_text())['parameter_sha256'])
        if status['status'] != 'complete' or [r['epoch'] for r in rows] != list(range(1, 201)):
            raise ValueError(method + ': need all 200 completed epochs, not a partial/best-epoch result.')
        final = rows[-1]
        if final['method'] != method or final['seed'] != 0 or final['test_examples'] != 10000:
            raise ValueError('Wrong method, seed or evaluation sample count.')
        error = float(final['test_error_pct'])
        if not math.isfinite(error) or not 0 <= error <= 100:
            raise ValueError('Invalid test error.')
        errors[method] = error
    if identities[0] != identities[1] or initializations[0] != initializations[1]:
        raise ValueError('Runs do not share the same configuration, worker and initial weights.')
    for method, reference in [('SAM-200', 2.7), ('SGD-200', 3.5)]:
        print('{}: epoch-200 test error {:.3f}%; paper reference {:.1f}%; gap {:+.3f} pp'.format(
            method, errors[method], reference, errors[method] - reference))
    print('Paired SGD-200 minus SAM: {:+.3f} percentage points.'.format(errors['SGD-200'] - errors['SAM-200']))
    print('One seed: no SD or confidence interval. Paper SGD selects the better of 200/400 epochs.')
    print('This first comparison is not the full multi-seed paper reproduction.')


if __name__ == '__main__':
    main()
