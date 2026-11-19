#!/usr/bin/env python3
"""Read-only, stdlib-only descriptive summaries; no hypothesis decision."""
import csv
import json
from pathlib import Path
import statistics
import sys


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def report(study):
    rows = []
    missing = []
    for seed in (0, 1, 2):
        for method in ('SGD', 'SAM', 'IGR'):
            path = study/('final_'+method+'_seed'+str(seed))/'metrics.json'
            if not path.exists(): missing.append(str(path.parent.name)); continue
            row = json.loads(path.read_text())
            assert row['status']=='complete' and row['method']==method and row['seed']==seed
            rows.append(row)
    write_csv(study/'per_seed_metrics.csv', rows)
    summary = []
    metric_names = ('clean_test_accuracy', 'test_ce', 'input_gradient_l2', 'input_gradient_squared_l2',
                    'sharpness_rho_0.025', 'sharpness_rho_0.05', 'weight_l2', 'probe_clean_accuracy', 'pgd_accuracy')
    for method in ('SGD', 'SAM', 'IGR'):
        group = [row for row in rows if row['method']==method]
        if len(group)!=3: continue
        for metric in metric_names:
            values = [row[metric] for row in group]
            summary.append(dict(method=method, metric=metric, n=3, mean=statistics.mean(values),
                                sample_std=statistics.stdev(values)))
    write_csv(study/'mean_std_metrics.csv', summary)
    paired = []
    mapping = {(row['method'], row['seed']): row for row in rows}
    if not missing:
        for method, baseline in (('SAM', 'SGD'), ('IGR', 'SGD'), ('IGR', 'SAM')):
            for metric in metric_names:
                diff = [mapping[method, seed][metric]-mapping[baseline, seed][metric] for seed in (0, 1, 2)]
                mean = statistics.mean(diff); sd = statistics.stdev(diff)
                # Marginal t interval, df=2; not adjusted for multiple comparisons.
                half_width = 4.302652729911275*sd/(3**.5)
                paired.append(dict(method=method, baseline=baseline, metric=metric, n=3,
                    mean_difference=mean, sample_std_difference=sd,
                    marginal_t95_low=mean-half_width, marginal_t95_high=mean+half_width))
    write_csv(study/'paired_differences.csv', paired)
    output = dict(complete=len(rows)==9, completed_evaluations=len(rows), missing=missing,
        caveats=['Three seeds; uncertainty estimates have low precision.',
                 'Marginal paired t intervals assume independent seed differences and are not multiplicity adjusted.',
                 'No equivalence margin was specified; similar point estimates do not establish equivalence.',
                 'Sharpness is an absolute-L2 parameterization-sensitive optimization lower-bound proxy.',
                 'PGD-20 with three restarts can overestimate robustness; no masking certificate.',
                 'Cross-method associations do not identify the causal mechanism inside SAM.',
                 'No automated hypothesis conclusion. Preserve all tuning, failed and resumed runs.'])
    (study/'summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    if summary:
        print('\nDescriptive means and sample standard deviations:')
        for row in summary:
            print('{method:3s} {metric:28s} {mean:.6f} +/- {sample_std:.6f}'.format(**row))


if __name__=='__main__': report(Path(sys.argv[1]).resolve())
