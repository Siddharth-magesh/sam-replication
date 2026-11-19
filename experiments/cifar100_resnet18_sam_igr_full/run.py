#!/usr/bin/env python3
"""Full CIFAR-100 study. Dry run by default; use --execute on the prepared A100."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REVISION = '7f34662deec03341b78cccd4ce18ec2f7900429b'
METHODS = ('SGD', 'SAM', 'IGR')
SEEDS = (0, 1, 2)
LAMBDAS = (0., .01, .1)


def read(path): return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projection():
    # Supplied one-epoch measurements, including per-epoch validation/checkpoint costs.
    # Startup included once per trial; repeated compilation in epoch extrapolation is conservative.
    times = {'SGD': (17.467821861, .3275722, .3046234, 41.5198753),
             'SAM': (37.363633532, .275924865, .3054479, 52.7172141),
             'IGR': (39.528113408, .250972776, .306870011, 54.2016171)}
    final = {m: (sum(times[m][:3])*100+times[m][3])*3/3600 for m in METHODS}
    tuning = final['IGR']
    return dict(source='User-supplied A100 timing 20261119_153717_b8692b; estimates, not full-run measurements',
                final_training_hours=final, tuning_hours=tuning, unmeasured_evaluation_reserve_hours=1.,
                allowance=.2, total_estimated_hours=(sum(final.values())+tuning+1)*1.2)


def config(root, output, gpu, method, seed, lam, stage, job):
    return dict(protocol='cifar100_resnet18_sam_igr_full_v1', epochs=100, train_per_class=450,
        batch_size=128, seed=seed, split_seed=2026, lr=.1, momentum=.9, weight_decay=.0005,
        clip=5., rho=.05, igr_lambda=lam, method=method, stage=stage, job=job,
        worker_seconds=24*3600, repo=str(root/'work/sam'), output=str(output),
        data_dir=str(root/'tensorflow_datasets'), source_revision=REVISION, gpu=gpu,
        probe_seed=31415, pgd_epsilon=8/255, pgd_alpha=2/255, pgd_steps=20, pgd_restarts=3,
        sharp_radii=[.025, .05], sharp_steps=10, sharp_restarts=3)


def jobs():
    return [('tune_lambda_'+str(lam), 'IGR', 42, lam, 'tune') for lam in LAMBDAS]


def select_lambda(study):
    tuning = []
    for job, method, seed, lam, stage in jobs():
        row = read(study/job/'train_result.json')
        assert row['status']=='complete' and row['epochs']==100 and row['seed']==42
        assert row['method']=='IGR' and row['igr_lambda']==lam
        tuning.append(dict(lam=lam, validation_accuracy=row['validation']['accuracy'],
                           validation_ce=row['validation']['ce']))
    assert all(math.isfinite(r['validation_accuracy']) for r in tuning)
    selected = sorted(tuning, key=lambda r: (-r['validation_accuracy'], r['lam']))[0]['lam']
    decision = dict(selected_lambda=selected, tuning=tuning, test_used=False,
                    rule='Highest final-epoch validation accuracy; ties smallest lambda')
    path = study/'lambda_selection.json'
    if path.exists(): assert read(path)==decision, 'Frozen lambda selection differs'
    else: atomic(path, decision)
    return selected


def launch(ns, study, root, gpu, job, method, seed, lam, stage, phase, deadline):
    out = study/job; out.mkdir(exist_ok=True)
    cfg = config(root, out, gpu, method, seed, lam, stage, job)
    path = out/'config.json'
    if path.exists(): assert read(path)==cfg, 'Job configuration changed'
    else: atomic(path, cfg)
    marker = out/('train_result.json' if phase=='train' else 'metrics.json')
    if marker.exists():
        row = read(marker)
        assert row['status']=='complete' and row['method']==method and row['seed']==seed
        if phase=='train':
            assert row['epochs']==100 and row['igr_lambda']==lam
            assert digest(out/'latest.msgpack')==row['checkpoint_sha256'], 'Checkpoint changed'
        print('Reusing completed', job, phase, flush=True)
        return
    atomic(study/'progress.json', dict(job=job, phase=phase, status='running'))
    ns['command']([ns['PYTHON'], '-u', study/'worker.py', path, phase],
                   out/(phase+'.log'), ns['environment'](), deadline)
    assert marker.exists() and read(marker)['status']=='complete'


def export(study):
    subprocess.run([sys.executable, '-B', str(study/'report.py'), str(study)], check=True, timeout=120)
    archive = study.parent/(study.name+'_evidence.zip')
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(study.rglob('*')):
            if path.is_file() and path.suffix not in ('.msgpack', '.tmp', '.zip'):
                bundle.write(path, str(Path(study.name)/path.relative_to(study)))
    temporary.replace(archive)
    print('RESULTS:', study, '\nDOWNLOAD:', archive, '\nInstance remains active.', flush=True)



def validate_ledger(ledger):
    assert math.isfinite(ledger['charged_hours']) and ledger['charged_hours'] >= 0
    settled = [r for r in ledger['invocations'] if r['status'] != 'reserved']
    assert all(r['status'] in ('complete', 'failed', 'manually_settled') for r in settled)
    assert all(math.isfinite(r['elapsed_hours']) and r['elapsed_hours'] >= 0 for r in settled)
    assert abs(sum(r['elapsed_hours'] for r in settled)-ledger['charged_hours']) < 1e-8
    pending = [r for r in ledger['invocations'] if r['status'] == 'reserved']
    assert len(pending) <= 1
    assert all(math.isfinite(r['reserved_hours']) and r['reserved_hours'] > 0 for r in pending)
    return pending


def settle_receipt(ledger, hours):
    pending = validate_ledger(ledger)
    assert len(pending) == 1, 'Expected exactly one crashed reservation'
    assert math.isfinite(hours) and 0 <= hours <= pending[0]['reserved_hours']
    pending[0].update(status='manually_settled', elapsed_hours=hours)
    ledger['charged_hours'] += hours
    validate_ledger(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--work-root', type=Path, default=Path('/workspace/sam_igr_a100_timing'))
    parser.add_argument('--reuse-env', type=Path, default=Path('/workspace/sam_paper_replication/envs/sam'))
    parser.add_argument('--study-dir', type=Path, help='Existing external study directory to resume')
    parser.add_argument('--budget-hours', type=float, default=20.,
                        help='Cumulative incremental study cap, including failed invocations; frozen at creation')
    parser.add_argument('--settle-interrupted-hours', type=float,
                        help='Replace a crashed invocation reservation with independently verified actual hours; no training')
    args = parser.parse_args()
    assert math.isfinite(args.budget_hours) and args.budget_hours>0
    sources = {name: (HERE/(name+'.py')).read_text() for name in ('run', 'worker', 'runtime', 'report')}
    for name, source in sources.items(): ast.parse(source, filename=name, feature_version=(3, 8))
    print(json.dumps(projection(), indent=2))
    if not args.execute:
        print('PLAN ONLY. 3 development trials + 9 final runs, 100 epochs each; seeds 0/1/2.')
        print('Validation-only lambda selection; final test evaluation; cumulative cap:', args.budget_hours, 'h.')
        print('Run on prepared csci567-sam with --execute; no installation or GPU work performed.')
        return
    assert sys.platform=='linux', 'Use prepared Linux A100'
    assert subprocess.check_output(['hostname'], text=True, timeout=10).strip()=='csci567-sam'
    root = args.work_root.resolve()
    assert str(root).startswith('/workspace/'), 'Work root must be below /workspace'
    assert Path('/workspace').is_mount(), 'Persistent /workspace must be mounted'
    assert root != Path('/workspace/sam_paper_replication')
    import fcntl
    with ExitStack() as stack:
        lock_paths = [Path.home()/'.sam_igr_a100_timing.lock']
        old = Path('/workspace/sam_paper_replication/launcher.lock')
        if old.parent.exists(): lock_paths.append(old)
        for path in lock_paths:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip(), 'GPU busy'
        gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        assert 'A100' in gpu and float(gpu.rsplit(',', 1)[1])>=38000
        parent = root/'full_runs'; parent.mkdir(parents=True, exist_ok=True)
        if args.study_dir:
            study = args.study_dir.resolve()
            assert study.parent==parent and study.is_dir(), 'Resume only an existing full_runs study'
        else:
            assert args.settle_interrupted_hours is None
            assert args.budget_hours>=projection()['total_estimated_hours'], 'Cap below projected initial study cost'
            study = parent/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            study.mkdir()
        print('STUDY:', study, flush=True)
        manifest = dict(protocol='cifar100_resnet18_sam_igr_full_v1', root=str(root),
            prefix=str(args.reuse_env.resolve()), source_revision=REVISION, gpu=gpu,
            budget_hours=args.budget_hours,
            sources={name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()})
        if (study/'manifest.json').exists(): assert read(study/'manifest.json')==manifest, 'Frozen manifest mismatch'
        else:
            atomic(study/'manifest.json', manifest)
            for name, value in sources.items(): (study/(name+'.py')).write_text(value)
            atomic(study/'planning.json', projection())
        for name, sha in manifest['sources'].items():
            assert digest(study/(name+'.py'))==sha, 'Saved source snapshot changed'
        ledger_path = study/'incremental_budget.json'
        ledger = read(ledger_path) if ledger_path.exists() else dict(charged_hours=0., invocations=[])
        pending = validate_ledger(ledger)
        if args.settle_interrupted_hours is not None:
            settle_receipt(ledger, args.settle_interrupted_hours)
            atomic(ledger_path, ledger)
            print('Settled once. No training launched; preserve external timing evidence.')
            return
        assert not pending, 'Crash reservation pending. Verify actual elapsed hours, then settle explicitly before resume.'
        remaining = args.budget_hours-ledger['charged_hours']
        assert remaining>1/60, 'Study cumulative budget exhausted'
        receipt = dict(id=uuid.uuid4().hex, status='reserved', reserved_hours=remaining,
                       started_utc=datetime.now(timezone.utc).isoformat())
        ledger['invocations'].append(receipt); atomic(ledger_path, ledger)
        start = time.monotonic(); deadline = start+remaining*3600-20
        ns = dict(WORK_ROOT=str(root), REUSE_PREFIX=str(args.reuse_env.resolve()))
        status = 'failed'
        def interrupted(signum, frame): raise KeyboardInterrupt('Stop requested')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            exec(compile(sources['runtime'], str(HERE/'runtime.py'), 'exec'), ns)
            assert ns['PYTHON'].is_file() and ns['pinned_cuda_ready'](), 'Use timing environment; no installation performed'
            assert (root/'cuda/nvvm/libdevice/libdevice.10.bc').is_file()
            revision = subprocess.check_output(['git', '-C', str(ns['REPO']), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
            assert revision==REVISION
            assert not subprocess.check_output(['git', '-C', str(ns['REPO']), 'status', '--porcelain',
                '--untracked-files=no'], text=True, timeout=15).strip(), 'Runtime tracked source modified'
            ns['command']([ns['PYTHON'], '-c', ns['GPU_CHECK']], study/'runtime_check.log', ns['environment'](), deadline)
            for job, method, seed, lam, stage in jobs():
                launch(ns, study, root, gpu, job, method, seed, lam, stage, 'train', deadline)
            selected = select_lambda(study)
            final = [('final_'+m+'_seed'+str(s), m, s, selected if m=='IGR' else 0., 'final')
                     for s in SEEDS for m in METHODS]
            for job, method, seed, lam, stage in final:
                launch(ns, study, root, gpu, job, method, seed, lam, stage, 'train', deadline)
            for seed in SEEDS:
                rows = [read(study/('final_'+m+'_seed'+str(seed))/'train_result.json') for m in METHODS]
                assert len({r['initial_model_bn_sha256'] for r in rows})==1, 'Unpaired initialization'
                assert len({r['epoch0_order_sha256'] for r in rows})==1, 'Unpaired data order'
            for job, method, seed, lam, stage in final:
                launch(ns, study, root, gpu, job, method, seed, lam, stage, 'evaluate', deadline)
            status = 'complete'
        finally:
            signal.signal(signal.SIGTERM, previous)
            receipt.update(status=status, elapsed_hours=(time.monotonic()-start)/3600)
            ledger['charged_hours'] += receipt['elapsed_hours']; atomic(ledger_path, ledger)
            atomic(study/'status.json', dict(status=status, charged_hours=ledger['charged_hours'],
                note='Incremental active invocation time only; reconcile once with team ledger. Idle VM time excluded.'))
            export(study)


if __name__=='__main__': main()
