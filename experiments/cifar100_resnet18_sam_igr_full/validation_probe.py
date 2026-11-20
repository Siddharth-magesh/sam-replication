#!/usr/bin/env python3
"""Compare saved IGR tuning checkpoints on validation only; dry-run by default."""
import argparse
import ast
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from run import atomic, read, digest, validate_ledger

HERE = Path(__file__).resolve().parent
JOBS = ('tune_lambda_0.0', 'tune_lambda_0.01', 'tune_lambda_0.1')
PROTOCOL = 'cifar100_resnet18_sam_igr_full_v3_centered_bn'


def validate_jobs(study, manifest):
    assert manifest['protocol']==PROTOCOL, 'This probe requires the completed v3 protocol'
    for name in ('worker', 'runtime'):
        assert digest(study/(name+'.py'))==manifest['sources'][name], 'Saved source changed'
    configs = []
    for job, lam in zip(JOBS, (0., .01, .1)):
        folder = study/job; cfg = read(folder/'config.json'); trained = read(folder/'train_result.json')
        assert cfg['protocol']==PROTOCOL and cfg['method']=='IGR' and cfg['stage']=='tune'
        assert cfg['seed']==42 and cfg['epochs']==100 and cfg['igr_lambda']==lam
        assert cfg['bn_implementation']=='centered' and cfg['bn_calibration_batches']==100
        assert trained['status']=='complete' and trained['epochs']==100
        assert trained['seed']==42 and trained['method']=='IGR' and trained['igr_lambda']==lam
        for filename, key in (('latest.msgpack','checkpoint_sha256'),
                              ('evaluation_state.msgpack','evaluation_state_sha256')):
            assert digest(folder/filename)==trained[key], 'Missing/modified checkpoint: '+str(folder/filename)
        configs.append(cfg)
    # Only the method's regularization strength and job/output paths may differ.
    exclude = {'igr_lambda', 'job', 'output'}
    shared = [{k:v for k,v in cfg.items() if k not in exclude} for cfg in configs]
    assert shared[0]==shared[1]==shared[2], 'Tuning settings differ beyond lambda'
    return configs


def summarize(out):
    rows = [read(out/job/'metrics.json') for job in JOBS]
    assert all(r['status']=='complete' and r['n']==1000 and r['seed']==42 for r in rows)
    assert len({r['validation_probe_sha256'] for r in rows})==1, 'Validation subsets differ'
    assert [r['igr_lambda'] for r in rows]==[0., .01, .1]
    for row in rows:
        row['accuracy_delta_vs_lambda0_pp'] = row['validation_accuracy']-rows[0]['validation_accuracy']
        for metric in ('input_gradient_l2', 'input_gradient_squared_l2'):
            assert math.isfinite(row[metric]) and row[metric]>=0
            row[metric+'_ratio_vs_lambda0'] = row[metric]/rows[0][metric] if rows[0][metric]>0 else None
    fields = ('igr_lambda', 'seed', 'n', 'validation_accuracy', 'validation_ce',
              'accuracy_delta_vs_lambda0_pp', 'input_gradient_l2', 'input_gradient_squared_l2',
              'input_gradient_l2_ratio_vs_lambda0', 'input_gradient_squared_l2_ratio_vs_lambda0')
    with (out/'validation_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    atomic(out/'comparison.json', dict(status='complete', rows=rows,
        note='Exploratory, single development seed; same validation subset and calibrated BN. Not a causal or multi-seed hypothesis test.'))
    print('\nlambda | validation accuracy (%) | mean gradient L2 | mean squared L2', flush=True)
    for row in rows:
        print('{igr_lambda:g} | {validation_accuracy:.2f} | {input_gradient_l2:.6f} | {input_gradient_squared_l2:.6f}'.format(**row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    ast.parse((HERE/'validation_worker.py').read_text(), feature_version=(3, 8))
    if not args.execute:
        print('PLAN ONLY: saved epoch-100 IGR tuning checkpoints, lambda 0/0.01/0.1, development seed 42.')
        print('Same balanced 1,000 validation images; saved calibrated BN; accuracy and per-image input CE gradients.')
        print('No training, test images, re-selection, package installation or VM launch. Shared 15-minute cap.')
        print('Study:', args.study_dir)
        return
    assert sys.platform=='linux'
    assert subprocess.check_output(['hostname'], text=True, timeout=15).strip()=='csci567-sam'
    study = args.study_dir.resolve(); manifest = read(study/'manifest.json')
    root = Path(manifest['root']).resolve()
    assert str(root).startswith('/workspace/') and study.parent==root/'full_runs'
    assert Path('/workspace').is_mount()
    assert read(study/'status.json')['status']=='complete', 'Use a completed study'
    import fcntl
    with ExitStack() as stack:
        paths = [Path.home()/'.sam_igr_a100_timing.lock']
        previous = Path('/workspace/sam_paper_replication/launcher.lock')
        if previous.parent.exists(): paths.append(previous)
        for path in paths:
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip(), 'GPU busy'
        gpu = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        assert gpu==manifest['gpu'] and 'A100' in gpu
        configs = validate_jobs(study, manifest)
        # Freeze source job evidence; the operational ledger is the only original file updated.
        files = [study/'lambda_selection.json', study/'manifest.json', study/'status.json']
        files += [study/job/name for job in JOBS for name in
                  ('config.json','train_result.json','history.json','latest.msgpack','evaluation_state.msgpack')]
        fingerprints = {str(path):digest(path) for path in files}
        ledger_path = study/'incremental_budget.json'; ledger = read(ledger_path)
        assert not validate_ledger(ledger), 'Unsettled reservation; use existing settlement instructions'
        allowance = min(.25, manifest['budget_hours']-ledger['charged_hours'])
        assert allowance>1/60, 'Insufficient remaining study budget'
        out = study/'validation_probes'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        out.mkdir(parents=True)
        for name in ('validation_worker.py', 'validation_probe.py'):
            (out/name).write_text((HERE/name).read_text())
        (out/'saved_training_worker.py').write_bytes((study/'worker.py').read_bytes())
        (out/'saved_runtime.py').write_bytes((study/'runtime.py').read_bytes())
        atomic(out/'provenance.json',dict(source_study=str(study),source_sha256=fingerprints,
            loader_sha256=digest(out/'validation_worker.py'),validation_probe_seed=271828,
            note='Post-hoc evaluation only; original results and lambda selection preserved.'))
        receipt = dict(id=uuid.uuid4().hex,status='reserved',reserved_hours=allowance,
            started_utc=datetime.now(timezone.utc).isoformat(),validation_probe=str(out))
        ledger['invocations'].append(receipt); atomic(ledger_path,ledger)
        begin=time.monotonic(); deadline=begin+allowance*3600-20; status='failed'
        def interrupted(signum,frame): raise KeyboardInterrupt('Stop requested')
        old_signal=signal.signal(signal.SIGTERM,interrupted)
        try:
            ns=dict(WORK_ROOT=str(root),REUSE_PREFIX=manifest['prefix'])
            exec(compile((study/'runtime.py').read_text(),str(study/'runtime.py'),'exec'),ns)
            assert ns['pinned_cuda_ready'](), 'Pinned runtime required; it was not modified'
            assert subprocess.check_output(['git','-C',str(ns['REPO']),'rev-parse','HEAD'],text=True,timeout=15).strip()==manifest['source_revision']
            assert not subprocess.check_output(['git','-C',str(ns['REPO']),'status','--porcelain',
                '--untracked-files=no'],text=True,timeout=15).strip(), 'Runtime tracked source changed'
            ns['command']([ns['PYTHON'],'-c',ns['GPU_CHECK']],out/'runtime_check.log',ns['environment'](),deadline)
            for job,cfg in zip(JOBS,configs):
                target=out/job; target.mkdir()
                probe_config=dict(cfg,output=str(target),source_job=str(study/job),
                    diagnostic_checkpoint=str(study/job/'latest.msgpack'),
                    saved_worker=str(study/'worker.py'),saved_worker_sha256=manifest['sources']['worker'],
                    validation_probe_seed=271828,worker_seconds=int(allowance*3600))
                atomic(target/'config.json',probe_config)
                ns['command']([ns['PYTHON'],'-u',out/'validation_worker.py',target/'config.json'],
                    target/'evaluate.log',ns['environment'](),deadline)
                assert read(target/'status.json')['status']=='complete'
            summarize(out)
            assert all(digest(Path(path))==sha for path,sha in fingerprints.items()), 'Original evidence changed'
            status='complete'
        finally:
            signal.signal(signal.SIGTERM,old_signal)
            receipt.update(status=status,elapsed_hours=(time.monotonic()-begin)/3600)
            ledger['charged_hours']+=receipt['elapsed_hours']
            validate_ledger(ledger); atomic(ledger_path,ledger)
            atomic(out/'status.json',dict(status=status,elapsed_hours=receipt['elapsed_hours'],
                study_charged_hours=ledger['charged_hours'],
                note='Diagnostic receipt appended once to study ledger; original training/test reports unchanged.'))
            archive = out.parent/(out.name+'_validation_probe.zip')
            temporary = archive.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(out.rglob('*')):
                    if path.is_file() and path.suffix not in ('.msgpack', '.tmp'):
                        bundle.write(path, str(Path(out.name)/path.relative_to(out)))
            temporary.replace(archive)
            print('VALIDATION PROBE RESULTS:',out,'\nDOWNLOAD:',archive,flush=True)
            print('Instance remains active. Preserve failed attempts; no test data or optimizer updates.',flush=True)


if __name__=='__main__': main()
