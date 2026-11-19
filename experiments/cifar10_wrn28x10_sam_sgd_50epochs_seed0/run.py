#!/usr/bin/env python3
"""Explicitly enabled reduced experiment; no installation or automatic shelving."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/sam_paper_replication')
REVISION = 'f7b9ff46ee0deab8004d5508baf68b7e7c3dad1a'
METHODS = ('SAM-50', 'SGD-50')
SESSION_TREE = 'runs_50epoch'
SESSION_POINTER = 'latest_50epoch_session.txt'
# Actual user-supplied Colab setup attempts, excluding the already-carried prior baseline.
COLAB_ATTEMPTS = (
    ('20261118_170159_9a51d2', 0.056612481114722224),
    ('20261118_171927_afb4f9', 0.057211381206388884),
    ('20261118_172501_943138', 0.04230758545333335),
)

def read(path):
    return json.loads(path.read_text())

def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))

def checked(args):
    return subprocess.check_output(list(map(str, args)), text=True, timeout=30).strip()

def reconcile_ledger():
    import fcntl
    path = ROOT/'gpu_budget.json'
    assert path.is_file(), 'Restore the existing authoritative gpu_budget.json; do not start a new ledger.'
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = read(path)
        assert data['schema'] == 1 and data['limit_seconds'] == 144000
        assert all(0 <= a['charged_seconds'] <= a['reserved_seconds'] for a in data['attempts'])
        assert len({a['id'] for a in data['attempts']}) == len(data['attempts'])
        assert all(a.get('state') in ('finished', 'budget_stopped') for a in data['attempts']), 'Unsettled reservation; inspect ledger before launching'
        for attempt, hours in COLAB_ATTEMPTS:
            identifier = 'colab-10epoch-'+attempt
            known = [a for a in data['attempts'] if a['id'] in (identifier, attempt)]
            if known:
                assert len(known) == 1 and abs(known[0]['charged_seconds']-hours*3600) < 1e-6
                continue
            seconds = hours*3600
            data['attempts'].append(dict(id=identifier, phase='external_setup', method='none', seed=0,
                session='colab_10epoch/'+attempt, state='finished', reserved_seconds=seconds,
                charged_seconds=seconds, observed_seconds=seconds, exit_code=1,
                evidence='User-supplied pilot_gpu_budget.json; setup failure, no training'))
        assert sum(a['charged_seconds'] for a in data['attempts']) <= 144000
        backup = ROOT/'gpu_budget.before_50epoch_import.json'
        if not backup.exists(): atomic(backup, read(path))
        atomic(path, data)

def configure(session):
    import launch as runner
    config = dict(mode='replication', deterministic_reductions=True, tf_deterministic_ops='0',
        cudnn_deterministic='1', budget_limit_hours=40.0, budget_file=str(ROOT/'gpu_budget.json'),
        revision=REVISION, repo=str(ROOT/'work/sam'), session=str(session),
        data_dir=str(ROOT/'tensorflow_datasets'), seeds=[0], logical_replicas=8,
        batch_size=256, subbatch_size=32, lr=.1, weight_decay=.0005, rho=.05,
        gradient_clipping=5.0, experiment='cifar10_wrn28x10_sam_sgd_50epochs_seed0',
        schedule_epochs=50, training_epochs=50)
    runner.CONFIG = config
    runner.WORK_ROOT = ROOT; runner.ENV_PREFIX = ROOT/'envs/sam'
    runner.PYTHON = runner.ENV_PREFIX/'bin/python'; runner.REPO = ROOT/'work/sam'
    runner.DATA_DIR = ROOT/'tensorflow_datasets'; runner.SESSION = session
    runner.BUDGET_FILE = ROOT/'gpu_budget.json'; runner.REPLICATION_BUDGET_HOURS = 40.0
    runner.WORKER_SOURCE = (HERE/'worker.py').read_text()
    runner.SUPERVISOR_SOURCE = (HERE/'supervisor.py').read_text()
    runner.configured_xla_flags = lambda: ('--xla_gpu_cuda_data_dir='+str(ROOT/'cuda')+
                                         ' --xla_gpu_deterministic_reductions=true')
    return runner

def estimate(runner):
    hours = 0.
    for method in METHODS:
        folder = runner.SESSION/'preflight'/(method+'_seed_0')
        report = read(folder/'preflight.json'); manifest = read(folder/'manifest.json')
        assert read(folder/'status.json')['status'] == 'complete' and report['status'] == 'passed'
        assert report['signature'] == manifest['signature']
        assert manifest['identity']['config'] == runner.CONFIG
        assert manifest['identity']['worker_sha256'] == hashlib.sha256(runner.WORKER_SOURCE.encode()).hexdigest()
        target = runner.SESSION/'train'/(method+'_seed_0')
        completed = 0
        if (target/'status.json').exists() and read(target/'status.json')['status'] == 'complete':
            completed = 50
        else:
            checkpoints = sorted((target/'checkpoints').glob('epoch_*'))
            if checkpoints: completed = int(read(checkpoints[-1]/'metadata.json')['completed_epoch'])
        assert 0 <= completed <= 50
        cost = 0 if completed == 50 else 1.2*(report['startup_and_diagnostics_seconds']+
                                            (50-completed)*report['planning_epoch_seconds'])/3600
        print(method, 'remaining estimated hours:', round(cost, 2), flush=True)
        hours += cost
    return hours

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true', help='Continue latest 50-epoch session, preserving checkpoints.')
    p.add_argument('--session', type=Path, help='Explicit session, requires --resume.')
    p.add_argument('--max-hours', type=float, default=18.0, help='Invocation wall cap, including preflight; not an estimate.')
    args = p.parse_args()
    assert 0 < args.max_hours <= 40 and (args.session is None or args.resume)
    for name in ('worker.py', 'supervisor.py', 'launch.py', 'report.py'):
        ast.parse((HERE/name).read_text(), feature_version=(3, 8))
    if not args.execute:
        print('PLAN: csci567-sam A100 only; '+', then '.join(METHODS)+'; seed 0; cosine over 50 epochs.')
        print('Reuse installed legacy stack/cache; strict preflight; shared 40-hour ledger; no auto-shelving.')
        print('No files, network, installation or GPU work. Add --execute to start.')
        return
    assert sys.platform == 'linux', 'Execute on csci567-sam Linux.'
    import fcntl
    with urllib.request.urlopen('http://169.254.169.254/openstack/latest/meta_data.json', timeout=5) as response:
        assert json.load(response)['uuid'].lower() == '195e6a8f-8ae1-45be-bcd7-7845b0b2419f', 'Wrong instance'
    assert ROOT.is_dir() and Path('/workspace').is_mount(), 'Persistent /workspace required'
    with (ROOT/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not checked(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']), 'GPU already in use'
        gpu = checked(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']).splitlines()
        assert len(gpu) == 1 and 'A100' in gpu[0] and float(gpu[0].split(',')[-1]) >= 30000, 'Full A100 required'
        for suffix in ('envs/sam/bin/python', 'envs/sam/bin/ptxas', 'cuda/nvvm/libdevice/libdevice.10.bc', 'work/sam/requirements.txt'):
            assert (ROOT/suffix).is_file(), 'Existing setup missing: '+suffix
        assert list((ROOT/'tensorflow_datasets/cifar10').glob('**/dataset_info.json')), 'Prepared CIFAR-10 cache missing'
        assert checked(['git', '-C', ROOT/'work/sam', 'rev-parse', 'HEAD']) == REVISION
        reconcile_ledger()
        pointer = ROOT/SESSION_POINTER
        if args.resume:
            session = args.session or Path(pointer.read_text().strip())
        else:
            session = ROOT/SESSION_TREE/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session = session.resolve()
        assert ROOT/SESSION_TREE in session.parents, 'Use only this experiment session tree'
        if args.resume: assert (session/'config.json').is_file(), 'Resume session missing'
        runner = configure(session)
        runner.prepare_session()
        pointer.write_text(str(session)+'\n')
        print('SESSION='+str(session), flush=True)
        print('Reusing installed environment/cache; NO installation. No automatic shelving.', flush=True)
        os.environ.update(TF_CUDNN_DETERMINISTIC='1', TF_DETERMINISTIC_OPS='0', CUDA_VISIBLE_DEVICES='0',
                          SAM_INVOCATION_STOP=str(time.monotonic()+args.max_hours*3600))
        def interrupted(signum, frame): raise KeyboardInterrupt('Stop requested')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            for method in ('SGD-50', 'SAM-50'):
                if method in METHODS:
                    runner.launch('preflight', method, 0)
            available = min(runner.budget_snapshot()['remaining_hours'],
                            (float(os.environ['SAM_INVOCATION_STOP'])-time.monotonic())/3600)
            predicted = estimate(runner)
            assert predicted <= available-30/3600, 'Estimated remaining pair exceeds cap/budget; no training started'
            for method in METHODS:
                runner.launch('train', method, 0, resume=args.resume)
            subprocess.run([str(runner.PYTHON), str(HERE/'report.py'), str(session)], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=120)
        finally:
            print('Instance remains active. Preserve results, then run js2 stop csci567-sam from WSL.', flush=True)

if __name__ == '__main__':
    main()
