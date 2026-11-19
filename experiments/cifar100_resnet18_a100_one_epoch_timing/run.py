#!/usr/bin/env python3
"""Sequential A100 timing only. No work without --execute; no full-study launch."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import ExitStack

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--work-root', type=Path, default=Path('/workspace/sam_igr_a100_timing'))
    parser.add_argument('--reuse-env', type=Path, help='Validate/reuse legacy env without installing into it')
    parser.add_argument('--planning-budget-hours', type=float, default=26.0812,
                        help='Scenario ceiling only; verify actual remaining allocation separately')
    args = parser.parse_args()
    assert args.planning_budget_hours > 0
    sources = {name: (HERE/(name+'.py')).read_text() for name in
               ('setup_runtime', 'prepare', 'launch', 'worker', 'report')}
    for name, source in sources.items(): ast.parse(source, filename=name, feature_version=(3, 8))
    if not args.execute:
        print('PLAN ONLY: A100 >=38,000 MiB; sequential SGD/SAM/IGR; one epoch each; 60-minute shared deadline.')
        print('45k train/5k validation, batch 128, seed 42, unchanged notebook-08 updates.')
        print('Root:', args.work_root, '| reuse env:', args.reuse_env, '| planning ceiling:', args.planning_budget_hours)
        return
    assert sys.platform == 'linux', 'Execute on Linux A100'
    assert subprocess.check_output(['hostname'], text=True, timeout=10).strip() == 'csci567-sam', 'Use only csci567-sam'
    assert str(args.work_root.resolve()).startswith('/workspace/'), 'Use persistent /workspace storage'
    assert Path('/workspace').is_mount(), '/workspace must be mounted'
    assert args.work_root.resolve() != Path('/workspace/sam_paper_replication'), 'Keep separate output/runtime source root'
    if args.reuse_env: assert (args.reuse_env/'bin/python').is_file(), 'Reuse environment missing'
    import fcntl
    with ExitStack() as stack:
        # Coordinate with the previous original-JAX launcher and other copies of this launcher.
        locks = [Path.home()/'.sam_igr_a100_timing.lock']
        old = Path('/workspace/sam_paper_replication/launcher.lock')
        if old.parent.exists(): locks.append(old)
        for path in locks:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        assert not active, 'GPU busy; leave the running experiment alone'
        ns = dict(WORK_ROOT=str(args.work_root.resolve()), REUSE_PREFIX=str(args.reuse_env) if args.reuse_env else None,
                  USE_DRIVE=False, MAX_SESSION_MINUTES=60, MEASURE_CHECKPOINT=True,
                  WORKER_SOURCE=sources['worker'], __name__='timing_launcher')
        start = time.monotonic(); status = 'failed'
        def interrupted(signum, frame): raise KeyboardInterrupt('Stop requested')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            for name in ('setup_runtime', 'prepare', 'launch'):
                exec(compile(sources[name], str(HERE/(name+'.py')), 'exec'), ns)
            for method in ('SGD', 'SAM', 'IGR'): ns['launch_timing'](method)
            status = 'complete'
        finally:
            signal.signal(signal.SIGTERM, previous)
            session = ns.get('SESSION')
            if session:
                atomic = ns['atomic_json']
                atomic(session/'invocation.json', dict(status=status, elapsed_hours=(time.monotonic()-start)/3600,
                    note='Separate incremental receipt including setup/data. Not imported into the old ledger; reconcile once. Idle allocation excluded.'))
                plan = dict(epochs=100,seeds=3,tuning_trials=3,tuning_epochs=100,tuning_seeds=1,
                            extra_evaluation_minutes=60,allowance=.2,budget_hours=args.planning_budget_hours)
                atomic(session/'planning_scenario.json', plan)
                report = session/'timing_report.py'; report.write_text(sources['report'])
                # Stdlib-only report works even when the pinned environment/setup failed.
                result = subprocess.run([sys.executable, str(report), str(session), str(session/'planning_scenario.json')],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
                (session/'report.log').write_text(result.stdout); print(result.stdout, flush=True)
                archive = session.parent/(session.name+'_timing.zip')
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                    for path in session.rglob('*'):
                        if path.is_file() and path.suffix not in ('.msgpack','.tmp'):
                            z.write(path,str(Path(session.name)/path.relative_to(session)))
                print('RESULTS:',session,'\nDOWNLOAD:',archive,'\nInstance remains active.',flush=True)
                if result.returncode and status=='complete': raise RuntimeError('Timing report failed; inspect report.log')


if __name__ == '__main__': main()
