#!/usr/bin/env python3
"""Descriptive exploratory extension report; references are reused, never new runs."""
import csv
import json
from pathlib import Path
import statistics
import sys

METRICS=('clean_test_accuracy','test_ce','input_gradient_l2','input_gradient_squared_l2',
         'sharpness_rho_0.025','sharpness_rho_0.05','weight_l2','probe_clean_accuracy','pgd_accuracy')


def write_csv(path,rows):
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def report(study):
    rows=[];missing=[]
    for seed in (0,1,2):
        for method in ('SGD','SAM','IGR'):
            folder=study/('final_'+method+'_seed'+str(seed));path=folder/'metrics.json'
            if not path.exists():missing.append(folder.name);continue
            row=json.loads(path.read_text());assert row['status']=='complete' and row['method']==method and row['seed']==seed
            row['origin']='reused reference' if method!='IGR' else 'new exploratory IGR'
            rows.append(row)
    write_csv(study/'per_seed_metrics.csv',rows)
    summary=[]
    for method in ('SGD','SAM','IGR'):
        group=[r for r in rows if r['method']==method]
        if len(group)!=3:continue
        for metric in METRICS:
            values=[r[metric] for r in group]
            summary.append(dict(method=method,metric=metric,n=3,mean=statistics.mean(values),
                                sample_std=statistics.stdev(values),origin=group[0]['origin']))
    write_csv(study/'mean_std_metrics.csv',summary)
    differences=[]
    if not missing:
        by={(r['method'],r['seed']):r for r in rows}
        for baseline in ('SGD','SAM'):
            for metric in METRICS:
                values=[by['IGR',s][metric]-by[baseline,s][metric] for s in (0,1,2)]
                mean=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729911275*sd/3**.5
                differences.append(dict(method='IGR',baseline=baseline,metric=metric,n=3,
                    mean_difference=mean,sample_std_difference=sd,marginal_t95_low=mean-half,marginal_t95_high=mean+half))
    write_csv(study/'paired_differences.csv',differences)
    tuning=[];subsets=set()
    for folder in sorted(study.glob('tune_lambda_*')):
        row=dict(job=folder.name)
        if (folder/'tuning_rejected.json').exists():
            row.update(json.loads((folder/'tuning_rejected.json').read_text()))
        elif (folder/'train_result.json').exists():
            t=json.loads((folder/'train_result.json').read_text())
            row.update(igr_lambda=t['igr_lambda'],status='training_complete',
                       full_validation_accuracy=t['validation']['accuracy'],full_validation_ce=t['validation']['ce'])
            if (folder/'validation_probe/metrics.json').exists():
                p=json.loads((folder/'validation_probe/metrics.json').read_text())
                subsets.add(p['validation_probe_sha256'])
                for key in ('validation_accuracy','validation_ce','input_gradient_l2','input_gradient_squared_l2'):
                    row['probe_'+key]=p[key]
        else:row['status']='incomplete'
        tuning.append(row)
    assert len(subsets)<=1,'Development validation subsets differ'
    write_csv(study/'tuning_diagnostics.csv',tuning)
    result=dict(complete=not missing and len(rows)==9,new_final_igr_evaluations=sum(r['method']=='IGR' for r in rows),
        reused_reference_evaluations=sum(r['method']!='IGR' for r in rows),missing=missing,
        selection=json.loads((study/'lambda_selection.json').read_text()) if (study/'lambda_selection.json').exists() else None,
        caveats=['Exploratory extension designed after seeing prior test results; not an independent confirmatory experiment.',
                 'Positive-only selection is predeclared for an active IGR intervention; it may underperform the lambda-zero reference.',
                 'Development tuning uses one seed; final IGR uses three paired seeds against reused SGD/SAM references.',
                 'Marginal paired t95 intervals have df=2, assume independent seed differences, and are not multiplicity adjusted.',
                 'No equivalence margin was specified; similar point estimates do not establish equivalence.',
                 'Lower input CE gradients do not guarantee prediction stability or adversarial robustness.',
                 'Sharpness is a parameterization-sensitive optimization lower-bound proxy; no causal mechanism conclusion.'])
    (study/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    for row in summary:
        if row['metric']=='clean_test_accuracy':print('{method}: {mean:.3f} +/- {sample_std:.3f}% ({origin})'.format(**row))


if __name__=='__main__':report(Path(sys.argv[1]).resolve())
