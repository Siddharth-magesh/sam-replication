#!/usr/bin/env python3
"""Run the reviewed notebook worker on csci567-sam; no execution by default."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

SERVER_ID = '195e6a8f-8ae1-45be-bcd7-7845b0b2419f'
SERVER_NAME = 'csci567-sam'
REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / 'repro_notebooks'
ARCHIVE_SHA256 = 'c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'


def cell(notebook, marker):
    data = json.loads((NOTEBOOKS / notebook).read_text())
    matches = [''.join(c['source']) for c in data['cells']
               if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    if len(matches) != 1:
        raise RuntimeError('Notebook layout changed; review marker: ' + marker)
    ast.parse(matches[0])
    return matches[0]


def checked(command, **kwargs):
    return subprocess.check_output(command, text=True, timeout=30, **kwargs).strip()


def validate_ledger(data):
    assert data['schema'] == 1 and data['limit_seconds'] == 144000
    assert len({a['id'] for a in data['attempts']}) == len(data['attempts'])
    assert all(0 <= a['charged_seconds'] <= a['reserved_seconds'] for a in data['attempts'])
    assert sum(a['charged_seconds'] for a in data['attempts']) <= 144000


def import_ledger(target, source):
    if source:
        incoming = json.loads(source.read_text())
        validate_ledger(incoming)
        if target.exists():
            current = json.loads(target.read_text())
            validate_ledger(current)
            known = {a['id']: a for a in current['attempts']}
            assert all(known.get(a['id']) == a for a in incoming['attempts']), 'Divergent budget ledgers.'
        else:
            with target.open('x') as stream:
                json.dump(incoming, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
    assert target.exists(), 'Import the authoritative Colab ledger with --ledger-import; do not reset usage.'
    validate_ledger(json.loads(target.read_text()))


def verify_machine():
    with urllib.request.urlopen('http://169.254.169.254/openstack/latest/meta_data.json', timeout=5) as response:
        metadata = json.load(response)
    assert metadata['uuid'].lower() == SERVER_ID, 'This runner is restricted to csci567-sam.'


def gpu_jobs():
    return checked(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])


def cloud_check(binary):
    data = json.loads(checked([binary, 'server', 'show', SERVER_ID, '-f', 'json', '-c', 'id', '-c', 'name']))
    assert data['id'] == SERVER_ID and data['name'] == SERVER_NAME, 'Cloud credentials target the wrong instance.'


def shelve(binary):
    # Do not interrupt a teammate's GPU job. Our workers have already exited.
    if gpu_jobs():
        raise RuntimeError('GPU process still active: refusing to shelve over a running job.')
    for attempt in range(3):
        try:
            cloud_check(binary)
            state = checked([binary, 'server', 'show', SERVER_ID, '-f', 'value', '-c', 'status'])
            if state.startswith('SHELVED'):
                return
            subprocess.run([binary, 'server', 'shelve', SERVER_ID], check=True, timeout=60)
            print('Shelving requested for csci567-sam. Verify SHELVED_OFFLOADED from WSL/Exosphere.', flush=True)
            return
        except (subprocess.SubprocessError, OSError):
            if attempt == 2:
                raise
            time.sleep(5)


def execute(args, root):
    main = '04_wrn28_10_cifar10_paper_replication.ipynb'
    namespace = {'__name__': '__main__'}
    exec(cell(main, "WORK_ROOT = Path.home()"), namespace)
    session = args.session.resolve() if args.session else root / 'runs' / time.strftime('%Y%m%d_%H%M%S')
    assert root in session.parents and session != root, 'Session must be inside WORK_ROOT.'
    session.mkdir(parents=True, exist_ok=True)
    namespace.update(WORK_ROOT=root, ENV_PREFIX=root / 'envs' / 'sam',
                     PYTHON=root / 'envs' / 'sam' / 'bin' / 'python',
                     REPO=root / 'work' / 'sam', SESSION=session,
                     DATA_DIR=root / 'tensorflow_datasets', BUDGET_FILE=root / 'gpu_budget.json',
                     RUN_SETUP=args.setup, RUN_PREFLIGHT=args.phase == 'preflight',
                     RUN_TRAINING=args.phase == 'train', RENDER_RESULTS=False,
                     RUN_SELECTION=args.selection, RESUME_EXISTING=args.resume,
                     RUN_DIAGNOSTIC_SMOKE=args.phase == 'smoke',
                     DIAGNOSTIC_ONLY=args.diagnostic_only)
    namespace['CONFIG']['mode'] = 'smoke' if args.phase == 'smoke' else 'replication'
    namespace['CONFIG'].update(budget_file=str(namespace['BUDGET_FILE']),
                              repo=str(namespace['REPO']), session=str(session),
                              data_dir=str(namespace['DATA_DIR']),
                              diagnostic_only=args.diagnostic_only)
    import_ledger(namespace['BUDGET_FILE'], args.ledger_import)
    if args.cifar_archive:
        assert hashlib.sha256(args.cifar_archive.read_bytes()).hexdigest() == ARCHIVE_SHA256
        destination = namespace['DATA_DIR'] / 'downloads' / 'manual' / 'cifar-10-binary.tar.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == ARCHIVE_SHA256
        else:
            shutil.copy2(args.cifar_archive, destination)
    print('SESSION=' + str(session), flush=True)
    if args.setup:
        # Bootstrap and workers now come from the single replication notebook.
        exec(cell(main, 'def setup_command(command, environment=None):'), namespace)
    exec(cell(main, 'WORKER_SOURCE ='), namespace)
    exec(cell(main, 'def budget_snapshot():'), namespace)
    exec(cell(main, 'def selected_runs():'), namespace)
    if args.phase in ('train', 'smoke'):
        exec(cell(main, 'if RUN_TRAINING:\n'), namespace)
    if args.phase in ('train', 'report'):
        render = cell(main, 'if RENDER_RESULTS:\n')
        render = render.replace('from IPython.display import display', 'display = print')
        script = ('from pathlib import Path\nimport json, hashlib\nimport matplotlib\n'
                  'matplotlib.use("Agg")\nRENDER_RESULTS=True\n'
                  'SESSION=Path(' + repr(str(session)) + ')\n'
                  'CONFIG=' + repr(namespace['CONFIG']) + '\n'
                  'SEEDS=' + repr(namespace['SEEDS']) + '\n'
                  'def budget_snapshot():\n    return ' + repr(namespace['budget_snapshot']()) + '\n'
                  'WORKER_SOURCE=' + repr(namespace['WORKER_SOURCE']) + '\n' + render)
        subprocess.run([str(namespace['PYTHON']), '-c', script], check=True)
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'smoke', 'train', 'report'])
    parser.add_argument('--execute', action='store_true', help='Required to install or run anything.')
    parser.add_argument('--setup', action='store_true', help='Install/repair pinned environment before preflight.')
    parser.add_argument('--work-root', type=Path, default=Path('/workspace/sam_paper_replication'))
    parser.add_argument('--session', type=Path, help='Existing session; mandatory for train/report.')
    parser.add_argument('--selection', default='SAM-200_seed_0', help='Exact run label or all (nine runs).')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--diagnostic-only', action='store_true',
                        help='Stop after one-replica diagnostics; preflight only.')
    parser.add_argument('--ledger-import', type=Path)
    parser.add_argument('--cifar-archive', type=Path)
    parser.add_argument('--max-hours', type=float, default=0.75, help='Wall limit including setup; default 45 minutes.')
    parser.add_argument('--openstack-bin', default='openstack', help='Separate cloud-control CLI, not the SAM environment.')
    parser.add_argument('--leave-active', action='store_true', help='Explicitly disable automatic shelving; manual shelving required.')
    args = parser.parse_args()
    assert 0 < args.max_hours <= 40
    assert not args.setup or args.phase == 'preflight', '--setup is only for preflight.'
    assert not args.diagnostic_only or args.phase == 'preflight', '--diagnostic-only is only for preflight.'
    assert args.phase == 'smoke' or args.phase == 'preflight' or args.session, 'Specify the preflight SESSION for train/report.'
    assert not args.resume or args.phase == 'train', '--resume is only for full training.'
    if args.phase == 'smoke':
        assert args.selection == 'all', 'The smoke phase always runs SGD-200 and SAM-200 for seed 0.'
        assert not args.setup and not args.diagnostic_only, 'Smoke does not combine with setup or numerical-only diagnostics.'
    if not args.execute:
        print('PLAN ONLY. Target: csci567-sam, full A100 40 GB; phase:', args.phase)
        print('No installation, network, GPU execution or shelving. Add --execute to run.')
        return
    verify_machine()
    root = args.work_root.resolve()
    assert Path('/workspace').is_mount() and Path('/workspace') in root.parents
    assert REPOSITORY != root and REPOSITORY not in root.parents, 'Keep outputs outside the code checkout.'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not gpu_jobs(), 'An existing GPU job is active; no work or shelving performed.'
        cloud_binary = shutil.which(args.openstack_bin)
        if not args.leave_active:
            assert cloud_binary, 'Install/configure a separate OpenStack CLI for auto-shelving, or explicitly use --leave-active.'
            cloud_check(cloud_binary)
        started = time.time()
        status = {'phase': args.phase, 'server': SERVER_NAME, 'started_unix': started, 'status': 'running'}
        receipt = root / ('launcher_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
        def interrupt(signum, frame):
            raise KeyboardInterrupt('Launcher deadline or termination signal')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGALRM, interrupt)
        signal.alarm(max(1, int(args.max_hours * 3600)))
        try:
            gpu = checked(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']).splitlines()
            assert len(gpu) == 1 and 'A100' in gpu[0] and int(gpu[0].split(',')[-1]) >= 30000, 'Requires full A100 (40 GB); current 10 GB slice is not approved for this workload.'
            status['session'] = str(execute(args, root))
            status['status'] = 'complete'
        except BaseException as error:
            status.update(status='failed_or_interrupted', error=str(error))
            raise
        finally:
            signal.alarm(0)
            status['wall_seconds'] = time.time() - started
            status['auto_shelve_requested'] = False
            receipt.write_text(json.dumps(status, indent=2))
            if not args.leave_active:
                try:
                    shelve(cloud_binary)
                    status['auto_shelve_requested'] = True
                except Exception as error:
                    status['shelve_error'] = str(error)
                    print('AUTO-SHELVE FAILED: use js2 stop csci567-sam immediately.', file=sys.stderr, flush=True)
                receipt.write_text(json.dumps(status, indent=2))
                if 'shelve_error' in status and status['status'] == 'complete':
                    raise RuntimeError('Work completed but automatic shelving failed; instance may still bill.')
            else:
                print('Instance still bills. Run js2 stop csci567-sam when no job is active.', flush=True)


if __name__ == '__main__':
    main()
