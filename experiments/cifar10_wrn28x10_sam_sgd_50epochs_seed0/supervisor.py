# Linux-only wall-clock budget supervisor. Does not import any ML packages.
import os, sys, json, time, uuid, signal, subprocess, fcntl, ctypes
from pathlib import Path
config_path, phase, method, seed, resume = sys.argv[1:]
C=json.loads(Path(config_path).read_text())
LIMIT=40*3600.0
assert C['budget_limit_hours']==40.0
ledger_path=Path(C['budget_file']); ledger_path.parent.mkdir(parents=True,exist_ok=True)

def atomic(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp),str(path))

def interrupt(signum,frame):
    raise KeyboardInterrupt('Supervisor interrupted')

signal.signal(signal.SIGTERM,interrupt)
with (ledger_path.with_suffix('.lock')).open('a+') as lock:
    # Concurrent launches fail; no stale lock-file deletion or automatic budget reset.
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(schema=1,limit_seconds=LIMIT,attempts=[])
    assert ledger['schema']==1 and ledger['limit_seconds']==LIMIT
    assert all(0<=e['charged_seconds']<=e['reserved_seconds'] for e in ledger['attempts'])
    used=sum(e['charged_seconds'] for e in ledger['attempts'])
    assert 0<=used<=LIMIT
    remaining=min(LIMIT-used, float(os.environ["SAM_INVOCATION_STOP"])-time.monotonic())
    if remaining<=30:
        raise RuntimeError('Cumulative 40-hour budget exhausted (30-second shutdown reserve).')
    start=time.monotonic(); deadline=start+remaining
    entry=dict(id=uuid.uuid4().hex,session=C['session'],phase=phase,method=method,seed=int(seed),
               started_unix=time.time(),state='reserved',reserved_seconds=remaining,charged_seconds=remaining)
    # Reserve ALL remaining time durably before spawning. A crashed supervisor never refunds it.
    ledger['attempts'].append(entry); atomic(ledger_path,ledger)
    process=None; code=1; budget_stopped=False; confirmed_exit=True
    parent_pid=os.getpid()
    def child_setup():
        os.setsid()
        libc=ctypes.CDLL(None)
        if libc.prctl(1,signal.SIGKILL)!=0 or os.getppid()!=parent_pid:
            os._exit(125)  # Linux parent-death signal: no orphan GPU worker.
    def stop_child():
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid,signal.SIGINT)
            process.wait(timeout=min(3,max(0.1,deadline-time.monotonic()-5)))
        except (subprocess.TimeoutExpired,ProcessLookupError):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
    try:
        env=os.environ.copy()
        env['SAM_STOP_MONOTONIC']=str(deadline-20)
        process=subprocess.Popen([sys.executable,'-u',str(Path(C['session'])/'runner.py'),
                   phase,config_path,method,seed,resume],env=env,preexec_fn=child_setup)
        try:
            code=process.wait(timeout=max(0.1,deadline-time.monotonic()-10))
        except subprocess.TimeoutExpired:
            budget_stopped=True; code=124; stop_child()
        # Cooperative deadline stops may exit before the outer watchdog.
        if code!=0 and time.monotonic()>=deadline-21:
            budget_stopped=True
    except KeyboardInterrupt:
        code=130
    finally:
        # Ignore further interrupts while killing/reaping the child and settling the ledger.
        signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            stop_child()
        except BaseException:
            confirmed_exit=False
        elapsed=time.monotonic()-start
        if confirmed_exit and (process is None or process.poll() is not None):
            entry.update(state='budget_stopped' if budget_stopped else 'finished',
                         charged_seconds=min(remaining,elapsed),observed_seconds=elapsed,exit_code=code,
                         finished_unix=time.time())
        else:
            entry.update(state='unresolved',exit_code=code,observed_seconds=elapsed)
            # Keep the entire reservation charged. No further launch is possible.
        atomic(ledger_path,ledger)
        run=Path(C['session'])/phase/(method+'_seed_'+seed)
        run.mkdir(parents=True,exist_ok=True)
        atomic(run/('budget_attempt_'+entry['id']+'.json'),entry)
        if budget_stopped:
            path=run/'status.json'
            status=json.loads(path.read_text()) if path.exists() else {}
            status.update(status='budget_stopped',budget_attempt=entry['id'])
            atomic(path,status)
        print('Cumulative charged GPU hours:',sum(e['charged_seconds'] for e in ledger['attempts'])/3600,
              '/ 40; attempt:',entry['state'],flush=True)
        if elapsed>remaining or not confirmed_exit:
            print('Shutdown/accounting anomaly: budget locked; inspect scheduler accounting.',flush=True)
            code=124
    sys.exit(code)
