

def project_budget(rows,plan):
    assert set(r['method'] for r in rows)=={'SGD','SAM','IGR'} and len(rows)==3
    assert all(r['status']=='complete' and r['epochs_measured']==1 and r['steps']==351 for r in rows)
    assert len(set(r['initial_model_bn_sha256'] for r in rows))==1, 'Initial states differ'
    assert len(set(r['epoch_order_sha256'] for r in rows))==1, 'Training orders differ'
    assert len(set(r['worker_sha256'] for r in rows))==1, 'Workers differ'
    assert len(set((r['gpu'],r['batch_size']) for r in rows))==1, 'Hardware/batch differs'
    assert plan['epochs']>0 and plan['seeds']>=3 and 0<=plan['allowance']<=1
    assert plan['tuning_trials']>=0 and plan['tuning_epochs']>=0 and plan['tuning_seeds']>=0
    costs=[]
    for r in rows:
        per_epoch=r['epoch_training_seconds']+r['validation_seconds']+r['checkpoint_seconds']
        one_run=r['fixed_worker_overhead_seconds']+plan['epochs']*per_epoch
        costs.append(dict(method=r['method'],projected_per_seed_hours=one_run/3600,
            projected_all_seeds_hours=plan['seeds']*one_run/3600))
    igr=next(r for r in rows if r['method']=='IGR')
    tuning=plan['tuning_trials']*plan['tuning_seeds']*(igr['fixed_worker_overhead_seconds']+
        plan['tuning_epochs']*(igr['epoch_training_seconds']+igr['validation_seconds']+igr['checkpoint_seconds']))
    main=sum(r['projected_all_seeds_hours'] for r in costs)
    base=main+tuning/3600+plan['extra_evaluation_minutes']/60
    return dict(estimated_not_measured=True,plan=plan,per_method=costs,
        projected_main_hours=main,projected_additional_igr_search_hours=tuning/3600,
        unmeasured_extra_evaluation_reserve_hours=plan['extra_evaluation_minutes']/60,
        projected_total_before_allowance_hours=base,projected_total_with_allowance_hours=base*(1+plan['allowance']),
        fits_declared_planning_budget=base*(1+plan['allowance'])<=plan['budget_hours'],
        scope='Epoch-1 extrapolation; same data/batch/FP32/update/validation/checkpoint protocol only. No automatic full training.')

import json,sys,csv
from pathlib import Path
p=Path(sys.argv[1]);plan=json.loads(Path(sys.argv[2]).read_text())
rows=[]
for method in ('SGD','SAM','IGR'):
    f=p/(method+'_result.json')
    if f.exists():rows.append(json.loads(f.read_text()))
if not rows:
    print('No completed epochs. Inspect setup/data/method logs; there is no measured budget yet.');sys.exit(0)
fields=['method','epoch_training_seconds','validation_seconds','checkpoint_seconds','fixed_worker_overhead_seconds','worker_wall_seconds']
with (p/'measured_epoch_times.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
for r in rows:
    print('%s: train %.2f s | validation %.2f s | checkpoint %.2f s | other worker overhead %.2f s | worker total %.2f s' %
        tuple(r[k] for k in fields))
    print('  step median %.4f s, p90 %.4f s; warm-up probe stable: %s' %
        (r['step_statistics_seconds']['median'],r['step_statistics_seconds']['p90'],r['warmup_probe_stable']))
if len(rows)!=3:
    print('INCOMPLETE: only %d/3 methods completed. No full-study projection generated.'%len(rows));sys.exit(0)
result=project_budget(rows,plan)
(p/'budget_projection.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print('\nPROJECTED full experiment (not measured):')
for r in result['per_method']:print(r['method'],round(r['projected_all_seeds_hours'],2),'hours across',plan['seeds'],'seeds')
print('Main comparison:',round(result['projected_main_hours'],2),'hours')
print('Additional IGR search:',round(result['projected_additional_igr_search_hours'],2),'hours')
print('Unmeasured additional evaluation reserve:',result['unmeasured_extra_evaluation_reserve_hours'],'hours')
print('Total with allowance:',round(result['projected_total_with_allowance_hours'],2),'hours')
print('Within declared planning budget:',result['fits_declared_planning_budget'])
print('Projection only. No full experiment was launched. One-epoch validation accuracy is not a scientific result.')

