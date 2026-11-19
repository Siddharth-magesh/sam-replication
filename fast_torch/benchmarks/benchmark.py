"""Stdlib-only supervisor: idle-device check before importing CUDA, hard TOTAL cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .benchmark_configs import CASES

REPO = Path(__file__).resolve().parents[2]

def idle():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    if result: raise RuntimeError('GPU busy; refusing to interfere with JAX or another job: '+result)
    return dict(checked_unix=time.time(), compute_processes=result)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true'); p.add_argument('--mode', choices=['benchmark', 'train'], default='benchmark')
    p.add_argument('--minutes', type=float, default=45)
    p.add_argument('--cases', nargs='+', choices=list(CASES), default=['reference'])
    p.add_argument('--methods', nargs='+', choices=['SAM', 'SGD'], default=['SAM', 'SGD'])
    p.add_argument('--init', type=Path, required=True); p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--approval', type=Path)
    p.add_argument('--config', type=Path, default=REPO/'fast_torch/configs/cifar10_50.json')
    p.add_argument('--resume', action='store_true'); p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    assert 0 < args.minutes <= (45 if args.mode == 'benchmark' else 2400)
    assert args.mode != 'benchmark' or not args.resume
    assert args.mode != 'train' or len(args.cases) == 1
    assert 1 <= args.steps <= 100
    assert REPO not in args.output.resolve().parents and args.output.resolve() != REPO, 'Output outside repo'
    if not args.execute:
        print('PLAN ONLY:', args.mode, args.methods, args.cases, 'total cap minutes:', args.minutes)
        return
    assert sys.platform == 'linux', 'GPU launch requires Linux A100; CPU validation is separate'
    assert args.init.is_file() and args.archive.is_file() and args.config.is_file()
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    # Shared across all fast_torch output folders on this user account. Do not touch JAX locks.
    with (Path.home()/'.fast_torch_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_check = idle()
        start = time.monotonic(); deadline = start+args.minutes*60
        attempts = []; receipt = args.output/('gpu_hours_'+str(time.time_ns())+'.json')
        def interrupted(signum, frame): raise KeyboardInterrupt('Supervisor interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        def save(status):
            value = dict(status=status, allocated_elapsed_hours=(time.monotonic()-start)/3600,
                         cap_minutes=args.minutes, initial_idle_check=initial_check, attempts=attempts,
                         note='Import this incremental elapsed time into project accounting once. No legacy ledger reset/write.')
            temp = receipt.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)); temp.replace(receipt)
        save('running')
        try:
            for method in args.methods:
                for case in args.cases:
                    check = idle(); remaining = deadline-time.monotonic()-10
                    if remaining < 1: raise TimeoutError('Total cap reached')
                    output = args.output/(method+'_'+case)
                    output.mkdir(exist_ok=True)
                    env = dict(os.environ, FAST_TORCH_SUPERVISED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                               CUDA_VISIBLE_DEVICES='0', TORCH_LOGS='graph_breaks,recompiles')
                    cmd = [sys.executable, '-u', '-m', 'fast_torch.training.engine', '--mode', args.mode,
                           '--method', method, '--case', case, '--init', str(args.init.resolve()),
                           '--archive', str(args.archive.resolve()), '--output', str(output.resolve()),
                           '--config', str(args.config.resolve()), '--steps', str(args.steps)]
                    if args.approval: cmd += ['--approval', str(args.approval.resolve())]
                    if args.resume: cmd += ['--resume']
                    entry = dict(method=method, case=case, idle_check=check, status='running')
                    attempts.append(entry); save('running')
                    print('Launching', method, case, '| log:', output/'worker.log', flush=True)
                    child = None; child_start = time.monotonic()
                    try:
                        with (output/'worker.log').open('a') as log:
                            # timeout provides a second watchdog if the Python supervisor disappears.
                            child = subprocess.Popen(['timeout', '--signal=INT', '--kill-after=5s', str(int(remaining))+'s']+cmd,
                                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO, start_new_session=True)
                            while child.poll() is None:
                                if time.monotonic() >= deadline-5: raise TimeoutError('Total cap reached')
                                # Actual timestamped device samples, including initialization and compilation.
                                sample = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=3)
                                with (output/'gpu_samples.csv').open('a') as f: f.write(sample.stdout)
                                time.sleep(1)
                            entry['exit_code'] = child.returncode
                            entry['status'] = 'complete' if child.returncode == 0 else 'failed'
                            if child.returncode: raise RuntimeError('Worker failed; inspect '+str(output/'worker.log'))
                    finally:
                        if child is not None and child.poll() is None:
                            os.killpg(child.pid, signal.SIGINT)
                            try: child.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=3)
                        entry['elapsed_seconds'] = time.monotonic()-child_start
                        save('running')
            save('complete')
        except BaseException:
            save('failed_or_interrupted'); raise
        finally:
            print('GPU-hours receipt:', receipt, '| machine remains active.', flush=True)

if __name__ == '__main__': main()
