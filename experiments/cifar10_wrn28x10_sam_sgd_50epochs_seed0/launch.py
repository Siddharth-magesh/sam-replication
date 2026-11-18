"""Reviewed notebook launch helpers; runtime values configured by run.py."""
from pathlib import Path
import os, json, hashlib, subprocess, time, signal

def budget_snapshot():
    assert REPLICATION_BUDGET_HOURS==40.0 and CONFIG['budget_limit_hours']==40.0
    if BUDGET_FILE.exists():
        data=json.loads(BUDGET_FILE.read_text())
        assert data['schema']==1 and data['limit_seconds']==40*3600
        assert all(0<=a['charged_seconds']<=a['reserved_seconds'] for a in data['attempts'])
        charged=sum(a['charged_seconds'] for a in data['attempts'])
        assert 0<=charged<=40*3600
    else:
        charged=0.0
    return dict(charged_hours=charged/3600,remaining_hours=40-charged/3600)

def prepare_session():
    assert PYTHON.is_file() and REPO.is_dir(), 'Complete setup or configure the pinned environment.'
    assert SESSION.resolve()!=REPO.resolve() and REPO.resolve() not in SESSION.resolve().parents
    SESSION.mkdir(parents=True,exist_ok=True)
    files=[('config.json',json.dumps(CONFIG,indent=2)+'\n'),('runner.py',WORKER_SOURCE),('supervisor.py',SUPERVISOR_SOURCE)]
    for name,text in files:
        path=SESSION/name
        if path.exists():
            assert path.read_text()==text, 'Session code/config changed: use a new session, keeping the SAME budget ledger.'
        else:
            path.write_text(text)
    return SESSION/'supervisor.py'

def launch(phase,label,seed,resume=False):
    supervisor=prepare_session()
    folder=SESSION/phase/(label+'_seed_'+str(seed))
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('status')=='complete':
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['identity']['config']==CONFIG
        assert manifest['identity']['worker_sha256']==hashlib.sha256(WORKER_SOURCE.encode()).hexdigest()
        print('Already complete; no launch:',folder)
        return
    assert budget_snapshot()['remaining_hours']>30/3600,'Cumulative budget exhausted.'
    environment=os.environ.copy()
    environment['TFDS_DATA_DIR']=str(DATA_DIR)
    environment['PYTHONPATH']=str(REPO.parent)
    environment['PYTHONHASHSEED']=str(seed)
    environment['CUDA_HOME']=str(ENV_PREFIX)
    environment['CUDA_PATH']=str(ENV_PREFIX)
    environment['PATH']=str(ENV_PREFIX/'bin')+':'+environment.get('PATH','')
    environment['LD_LIBRARY_PATH']=str(ENV_PREFIX/'lib')+':'+environment.get('LD_LIBRARY_PATH','')
    environment['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    environment.setdefault('CUDA_VISIBLE_DEVICES','0')
    environment['XLA_FLAGS']=configured_xla_flags()
    log=SESSION/(phase+'_'+label+'_seed_'+str(seed)+'.log')
    # Regular-file stdout lets the independent supervisor survive notebook disconnection.
    with log.open('a') as stream,log.open('r') as reader:
        reader.seek(0,2)
        process=subprocess.Popen([str(PYTHON),'-u',str(supervisor),str(SESSION/'config.json'),
                    phase,label,str(seed),str(resume).lower()],cwd=str(REPO.parent),env=environment,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                print(reader.read(),end=''); time.sleep(1)
            print(reader.read(),end='')
        except KeyboardInterrupt:
            # Do not kill the supervisor abruptly: it must reap the worker and settle accounting.
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
    print('Persistent budget:',budget_snapshot())
    if process.returncode!=0 and (folder/'diagnostics.json').exists():
        print('Saved diagnostic report: '+str(folder/'diagnostics.json'),flush=True)
        print((folder/'diagnostics.json').read_text(),flush=True)
    assert process.returncode==0,'Worker stopped: inspect '+str(log)+'. Last committed checkpoint remains available.'
