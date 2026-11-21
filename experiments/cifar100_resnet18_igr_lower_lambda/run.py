#!/usr/bin/env python3
"""Exploratory lower-lambda IGR extension. Prepare/execute only with --execute."""
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'cifar100_resnet18_sam_igr_full'
CORE_PATH = HERE/'core_run.py' if (HERE/'core_run.py').is_file() else ORIGINAL/'run.py'
spec = importlib.util.spec_from_file_location('full_study_core', CORE_PATH)
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
read, atomic, digest = core.read, core.atomic, core.digest
GRID = (1e-5, 1e-4, 1e-3)
SEEDS = (0, 1, 2)
EXTENSION = 'cifar100_igr_lower_lambda_exploratory_v1'
V3 = 'cifar100_resnet18_sam_igr_full_v3_centered_bn'


def job_name(lam): return 'tune_lambda_'+str(lam)


def select(study):
    candidates = []; rejected = []
    for lam in GRID:
        folder = study/job_name(lam)
        if (folder/'tuning_rejected.json').exists():
            rejected.append(read(folder/'tuning_rejected.json')); continue
        row = read(folder/'train_result.json')
        assert row['status']=='complete' and row['epochs']==100 and row['seed']==42
        assert row['method']=='IGR' and row['igr_lambda']==lam
        accuracy = row['validation']['accuracy']
        assert math.isfinite(accuracy)
        candidates.append(dict(lam=lam, validation_accuracy=accuracy, validation_ce=row['validation']['ce']))
    assert candidates, 'All positive candidates failed; no final model selected'
    best = min(candidates, key=lambda r: (-r['validation_accuracy'], r['lam']))
    control = read(study/'reference_lambda0.json')['validation']['accuracy']
    decision = dict(selected_lambda=best['lam'], tuning=candidates, rejected=rejected,
        reference_lambda0_validation_accuracy=control,
        selected_validation_delta_vs_lambda0_pp=best['validation_accuracy']-control,
        selected_beats_lambda0=best['validation_accuracy']>control,
        test_used=False, sensitivity_used=False,
        rule='Highest final-epoch full-validation accuracy among the three predeclared POSITIVE lambdas; ties smallest lambda. Lambda zero is an external reference, not a candidate.',
        note='Active IGR is selected by design for this exploratory extension; selection does not imply an improvement over the control.')
    path = study/'lambda_selection.json'
    if path.exists(): assert read(path)==decision, 'Frozen selection changed'
    else: atomic(path, decision)
    return best['lam']


def validate_reference(reference, sources):
    manifest = read(reference/'manifest.json')
    assert manifest['protocol']==V3 and read(reference/'status.json')['status']=='complete'
    for name in ('worker','runtime'):
        assert hashlib.sha256(sources[name].encode()).hexdigest()==manifest['sources'][name], 'Numerical/runtime source differs from completed v3'
    control = read(reference/'tune_lambda_0.0/train_result.json')
    assert control['seed']==42 and control['epochs']==100 and control['igr_lambda']==0 and control['method']=='IGR'
    assert control['status']=='complete'
    for seed in SEEDS:
        for method in ('SGD','SAM'):
            folder = reference/('final_'+method+'_seed'+str(seed))
            row = read(folder/'metrics.json'); trained = read(folder/'train_result.json')
            assert row['status']=='complete' and row['seed']==seed and row['method']==method
            assert trained['status']=='complete' and trained['epochs']==100
            assert trained['method']==method and trained['seed']==seed
            cfg = read(folder/'config.json')
            expected = core.config(Path(manifest['root']),folder,manifest['gpu'],method,seed,0.,'final',folder.name)
            assert cfg==expected, 'Reference configuration differs from preserved v3 protocol'
    return manifest


def snapshot_reference(reference, study):
    tracked = [reference/'manifest.json',reference/'lambda_selection.json',reference/'tune_lambda_0.0/train_result.json']
    for seed in SEEDS:
        for method in ('SGD','SAM'):
            name='final_'+method+'_seed'+str(seed); folder=study/name; folder.mkdir(exist_ok=True)
            for filename in ('metrics.json','train_result.json','config.json'):
                source=reference/name/filename; target=folder/filename
                if target.exists(): assert digest(target)==digest(source)
                else: target.write_bytes(source.read_bytes())
                tracked.append(source)
            atomic(folder/'reference_only.json',dict(source_job=str(reference/name),retrained=False))
    control=reference/'tune_lambda_0.0/train_result.json'
    if (study/'reference_lambda0.json').exists(): assert read(study/'reference_lambda0.json')==read(control)
    else: (study/'reference_lambda0.json').write_bytes(control.read_bytes())
    hashes={str(p):digest(p) for p in tracked}
    record=dict(source_study=str(reference),sha256=hashes,
        note='Completed reference SGD/SAM metrics reused; no reference checkpoint or test result modified. Prior test results have already been seen; new results are exploratory.')
    path=study/'reference_provenance.json'
    if path.exists(): assert read(path)==record, 'Reference evidence changed'
    else: atomic(path,record)


def run_job(ns,study,root,gpu,job,seed,lam,stage,phase,deadline):
    core.launch(ns,study,root,gpu,job,'IGR',seed,lam,stage,phase,deadline)


def validation_measure(ns,study,job,deadline):
    folder=study/job; target=folder/'validation_probe'
    if (target/'metrics.json').exists():
        row=read(target/'metrics.json')
        assert row['status']=='complete' and row['seed']==42 and row['igr_lambda']==read(folder/'config.json')['igr_lambda']
        return
    target.mkdir(exist_ok=True)
    cfg=dict(read(folder/'config.json'), output=str(target),source_job=str(folder),
        diagnostic_checkpoint=str(folder/'latest.msgpack'),saved_worker=str(study/'worker.py'),
        saved_worker_sha256=digest(study/'worker.py'),validation_probe_seed=271828)
    atomic(target/'config.json',cfg)
    ns['command']([ns['PYTHON'],'-u',study/'validation_worker.py',target/'config.json'],
        target/'evaluate.log',ns['environment'](),deadline)
    assert read(target/'metrics.json')['status']=='complete'


def export(study):
    subprocess.run([sys.executable,'-B',str(study/'report.py'),str(study)],check=True,timeout=120)
    archive=study.parent/(study.name+'_evidence.zip'); temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(study.rglob('*')):
            if p.is_file() and p.suffix not in ('.msgpack','.tmp'):
                z.write(p,str(Path(study.name)/p.relative_to(study)))
    temporary.replace(archive)
    print('RESULTS:',study,'\nDOWNLOAD:',archive,'\nInstance remains active.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-study',type=Path,default=Path('/workspace/sam_igr_a100_timing/full_runs/20261120_043937_4863bf'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--study-dir',type=Path,help='Exact extension directory to resume')
    parser.add_argument('--budget-hours',type=float,default=8.,help='Frozen incremental cap for this extension only; previous usage remains separate')
    parser.add_argument('--settle-interrupted-hours',type=float,help='Verified active hours of an uncatchably interrupted invocation; settle only')
    args=parser.parse_args()
    assert math.isfinite(args.budget_hours) and args.budget_hours>0
    source_dir=HERE if (HERE/'worker.py').exists() else ORIGINAL
    sources=dict(run=(HERE/'run.py').read_text(),core_run=CORE_PATH.read_text(),
        worker=(source_dir/'worker.py').read_text(),runtime=(source_dir/'runtime.py').read_text(),
        validation_worker=(source_dir/'validation_worker.py').read_text(),report=(HERE/'report.py').read_text())
    for name,source in sources.items():ast.parse(source,filename=name,feature_version=(3,8))
    if not args.execute:
        print('PLAN ONLY. Exploratory positive lambda grid:',GRID)
        print('Three 100-epoch tuning runs, development seed 42; best positive lambda by final validation accuracy, ties smaller.')
        print('Then fresh 100-epoch IGR runs on seeds 0/1/2, even if validation is worse than lambda zero.')
        print('Validation sensitivity is diagnostic only; final test occurs after selection and all three final training runs.')
        print('Reuse existing paired SGD/SAM references; no retraining of baselines. Incremental cap:',args.budget_hours,'h.')
        print('Planning estimate ~5.5 h: six ~40-minute active-IGR runs plus 0.5 h unmeasured evaluation reserve, with 20% allowance. Not measured for this grid.')
        return
    assert sys.platform=='linux'
    assert subprocess.check_output(['hostname'],text=True,timeout=15).strip()=='csci567-sam'
    reference=args.reference_study.resolve(); original=validate_reference(reference,sources)
    root=Path(original['root']).resolve()
    assert str(root).startswith('/workspace/') and reference.parent==root/'full_runs' and Path('/workspace').is_mount()
    import fcntl
    with ExitStack() as stack:
        paths=[Path.home()/'.sam_igr_a100_timing.lock']
        old=Path('/workspace/sam_paper_replication/launcher.lock')
        if old.parent.exists():paths.append(old)
        for path in paths:
            handle=stack.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip(),'GPU busy'
        gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        assert gpu==original['gpu'] and 'A100' in gpu
        parent=root/'lower_lambda_runs';parent.mkdir(exist_ok=True);pointer=root/'latest_lower_lambda_study.txt'
        if args.study_dir:
            study=args.study_dir.resolve();assert study.parent==parent and study.is_dir()
        else:
            assert args.settle_interrupted_hours is None
            if pointer.exists():
                previous=Path(pointer.read_text().strip())
                assert (previous/'status.json').exists() and read(previous/'status.json')['status']=='complete','Previous extension unfinished; resume its exact --study-dir'
            assert args.budget_hours>=5.5,'Initial cap below planning estimate'
            study=parent/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);study.mkdir()
        print('EXTENSION STUDY:',study,flush=True)
        manifest=dict(extension=EXTENSION,numerical_protocol=V3,reference_study=str(reference),gpu=gpu,
            root=str(root),prefix=original['prefix'],budget_hours=args.budget_hours,grid=list(GRID),seeds=list(SEEDS),
            sources={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()})
        if (study/'manifest.json').exists():assert read(study/'manifest.json')==manifest,'Frozen extension manifest mismatch'
        else:
            atomic(study/'manifest.json',manifest)
            for name,source in sources.items():(study/(name+'.py')).write_text(source)
        for name,sha in manifest['sources'].items():assert digest(study/(name+'.py'))==sha
        snapshot_reference(reference,study)
        pointer.write_text(str(study)+'\n')
        ledger_path=study/'incremental_budget.json'
        ledger=read(ledger_path) if ledger_path.exists() else dict(charged_hours=0.,invocations=[])
        pending=core.validate_ledger(ledger)
        if args.settle_interrupted_hours is not None:
            core.settle_receipt(ledger,args.settle_interrupted_hours);atomic(ledger_path,ledger)
            print('Interrupted receipt settled once; no training launched.');return
        assert not pending,'Unsettled crash reservation; verify active hours and settle explicitly'
        remaining=args.budget_hours-ledger['charged_hours'];assert remaining>1/60,'Extension budget exhausted'
        receipt=dict(id=uuid.uuid4().hex,status='reserved',reserved_hours=remaining,started_utc=datetime.now(timezone.utc).isoformat())
        ledger['invocations'].append(receipt);atomic(ledger_path,ledger)
        begin=time.monotonic();deadline=begin+remaining*3600-20;status='failed'
        def interrupted(signum,frame):raise KeyboardInterrupt('Stop requested')
        previous_signal=signal.signal(signal.SIGTERM,interrupted)
        try:
            ns=dict(WORK_ROOT=str(root),REUSE_PREFIX=original['prefix'])
            exec(compile(sources['runtime'],str(study/'runtime.py'),'exec'),ns)
            assert ns['pinned_cuda_ready']()
            assert subprocess.check_output(['git','-C',str(ns['REPO']),'rev-parse','HEAD'],text=True,timeout=15).strip()==original['source_revision']
            assert not subprocess.check_output(['git','-C',str(ns['REPO']),'status','--porcelain','--untracked-files=no'],text=True,timeout=15).strip()
            ns['command']([ns['PYTHON'],'-c',ns['GPU_CHECK']],study/'runtime_check.log',ns['environment'](),deadline)
            for lam in GRID:
                job=job_name(lam)
                if (study/job/'tuning_rejected.json').exists():
                    print('Preserving rejected tuning trial:',job,flush=True);continue
                try:run_job(ns,study,root,gpu,job,42,lam,'tune','train',deadline)
                except RuntimeError:
                    folder=study/job
                    if not ((folder/'train_status.json').exists() and read(folder/'train_status.json').get('error')=='Non-finite value' and (folder/'numerical_failure.json').exists()):raise
                    atomic(folder/'tuning_rejected.json',dict(lam=lam,status='rejected_nonfinite',evidence='numerical_failure.json'));continue
                validation_measure(ns,study,job,deadline)
            selected=select(study)
            for seed in SEEDS:run_job(ns,study,root,gpu,'final_IGR_seed'+str(seed),seed,selected,'final','train',deadline)
            for seed in SEEDS:
                rows=[read(study/('final_'+m+'_seed'+str(seed))/'train_result.json') for m in ('SGD','SAM','IGR')]
                assert len({r['initial_model_bn_sha256'] for r in rows})==1
                assert len({r['epoch0_order_sha256'] for r in rows})==1
            for seed in SEEDS:run_job(ns,study,root,gpu,'final_IGR_seed'+str(seed),seed,selected,'final','evaluate',deadline)
            expected=read(study/'reference_provenance.json')['sha256']
            assert all(digest(Path(path))==sha for path,sha in expected.items()),'Reference evidence changed'
            status='complete'
        finally:
            signal.signal(signal.SIGTERM,previous_signal)
            receipt.update(status=status,elapsed_hours=(time.monotonic()-begin)/3600)
            ledger['charged_hours']+=receipt['elapsed_hours'];core.validate_ledger(ledger);atomic(ledger_path,ledger)
            atomic(study/'status.json',dict(status=status,incremental_extension_hours=ledger['charged_hours'],
                note='New extension receipts only. Add once to previous team accounting; parent study journal unchanged, idle allocation excluded.'))
            export(study)


if __name__=='__main__':main()
