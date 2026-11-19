import os, sys, json, time, hashlib, subprocess, shutil, tarfile, io, urllib.request
import threading, signal, uuid, csv, traceback, platform, fcntl
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(WORK_ROOT)
DRIVE = Path('/content/drive/MyDrive/sam_replication')
SOURCE_REVISION = '7f34662deec03341b78cccd4ce18ec2f7900429b'
REPO = ROOT / 'work' / 'sam'; PREFIX = ROOT / 'envs' / 'sam'; PYTHON = PREFIX / 'bin' / 'python'
if REUSE_PREFIX:
    PREFIX = Path(REUSE_PREFIX).resolve(); PYTHON = PREFIX/'bin/python'
SESSION = None

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(str(tmp), str(path))

def command(args, log, environment=None, deadline=None, check=True):
    """Stream file logs; bound subprocesses even if the notebook kernel disappears."""
    args = list(map(str, args))
    # micromamba uses -c for channels; only Python's own -c contains code.
    python_inline = len(args) > 2 and Path(args[0]).name.startswith('python') and args[1] == '-c'
    if python_inline:
        compile(args[2], '<subprocess Python>', 'exec')
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    remaining = int((deadline or (time.monotonic()+1800))-time.monotonic())
    if remaining <= 0: raise TimeoutError('Session deadline reached; no new process started')
    bounded = ['timeout', '--signal=INT', '--kill-after=15s', str(remaining)+'s']+args
    label = list(args[:5])
    if python_inline:
        label[2] = '<embedded Python>'
    print('Running:', ' '.join(label), flush=True)
    with log.open('a') as output:
        output.write('\nCOMMAND '+repr(args)+'\n'); output.flush()
        offset = log.stat().st_size
        process = subprocess.Popen(bounded, stdout=output, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            with log.open() as reader:
                reader.seek(offset); heartbeat = time.monotonic()
                while process.poll() is None:
                    chunk = reader.read()
                    if chunk: print(chunk, end='', flush=True)
                    if time.monotonic()-heartbeat > 30:
                        print('[still running; log:', log.name+']', flush=True); heartbeat = time.monotonic()
                    time.sleep(1)
                print(reader.read(), end='', flush=True)
            if check and process.returncode: raise RuntimeError('Exit %s; see %s' % (process.returncode, log))
            return process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()

def environment():
    env = os.environ.copy()
    env.update(PATH=str(PREFIX/'bin')+':'+env.get('PATH', ''),
        LD_LIBRARY_PATH=str(PREFIX/'lib')+':'+env.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=str(REPO.parent), CUDA_HOME=str(PREFIX), CUDA_PATH=str(PREFIX),
        CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONHASHSEED='0',
        TFDS_DATA_DIR=str(ROOT/'tensorflow_datasets'), TF_DETERMINISTIC_OPS='0',
        TF_CUDNN_DETERMINISTIC='0',
        XLA_FLAGS='--xla_gpu_cuda_data_dir='+str(ROOT/'cuda')+
            ' --xla_gpu_deterministic_reductions='+'false')
    return env

VERSION_CHECK = '''
import sys
from importlib.metadata import version
assert sys.version_info[:2] == (3,8)
expected={'jax':'0.2.6','jaxlib':'0.1.57+cuda110','flax':'0.2.2','numpy':'1.18.5','tensorflow':'2.3.1'}
for package,want in expected.items():
    got=version(package); assert got==want,(package,got,want)
import numpy, tensorflow, flax, jax
print('Verified pinned packages:',expected,flush=True)
'''
GPU_CHECK = VERSION_CHECK + '''
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
x=jax.jit(lambda a:a@a)(jax.numpy.ones((2,2))); x.block_until_ready()
print('GPU initialization PASS:',jax.devices(),flush=True)
'''

def pinned_cuda_ready():
    packages = {}
    for path in (PREFIX/'conda-meta').glob('*.json'):
        record = json.loads(path.read_text())
        packages[record['name']] = record['version']
    return (packages.get('cudatoolkit', '').startswith('11.0.') and
            packages.get('cudnn', '').split('.')[:3] == ['8', '0', '5'] and packages.get('cuda-nvcc') == '11.3.122')

def setup(deadline):
    if not REPO.exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        command(['git', 'clone', '--branch', 'repro', 'https://github.com/Siddharth-magesh/sam-replication.git', REPO], SESSION/'setup.log', deadline=deadline)
        command(['git', '-C', REPO, 'checkout', '--detach', SOURCE_REVISION], SESSION/'setup.log', deadline=deadline)
    actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
    assert actual == SOURCE_REVISION, 'Existing runtime checkout differs; use a fresh ROOT.'
    assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True, timeout=30).strip(), 'Tracked source modifications'
    ready = False
    if PYTHON.exists() and (PREFIX/'bin'/'ptxas').exists() and pinned_cuda_ready():
        ready = command([PYTHON, '-c', VERSION_CHECK], SESSION/'environment_check.log', environment(), deadline, check=False) == 0
    if not ready and REUSE_PREFIX:
        raise RuntimeError('Existing legacy environment failed pin validation; it was NOT modified. Inspect environment_check.log.')
    if not ready:
        manager = ROOT/'bin'/'micromamba'
        if not manager.exists():
            with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest', timeout=120) as response:
                data = response.read()
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as archive:
                member = archive.getmember('bin/micromamba'); assert member.isfile()
                manager.parent.mkdir(parents=True, exist_ok=True)
                manager.write_bytes(archive.extractfile(member).read())
            manager.chmod(0o755)
        os.environ['MAMBA_ROOT_PREFIX'] = str(ROOT/'mamba')
        action = 'install' if PYTHON.exists() else 'create'
        command([manager, action, '-p', PREFIX, '-c', 'conda-forge', '--override-channels',
                 'python=3.8', 'pip=24.0', 'cudatoolkit=11.0', 'cudnn=8.0.5', '-y'], SESSION/'setup.log', deadline=deadline)
        command([manager, 'install', '-p', PREFIX, '-c', 'nvidia/label/cuda-11.3.1', '--override-channels',
                 'cuda-nvcc=11.3.122', '-y'], SESSION/'setup.log', deadline=deadline)
        lines = (REPO/'requirements.txt').read_text().splitlines()
        requirements = ROOT/'requirements_runtime.txt'
        requirements.write_text('\n'.join(s.replace('absl-py==0.11.0', 'absl-py==0.10.0')
            for s in lines if not s.startswith('jaxlib'))+'\n')
        command([PYTHON, '-m', 'pip', 'install', 'jaxlib==0.1.57+cuda110', '-f',
                 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'], SESSION/'setup.log', deadline=deadline)
        command([PYTHON, '-m', 'pip', 'install', '-r', requirements], SESSION/'setup.log', deadline=deadline)
    assert pinned_cuda_ready(), 'CUDA/cuDNN pins differ; inspect setup log instead of upgrading JAX'
    atomic_json(SESSION/'cuda_packages.json', [json.loads(p.read_text()) for p in (PREFIX/'conda-meta').glob('*.json')
        if p.name.startswith(('cudatoolkit-', 'cudnn-', 'cuda-nvcc-'))])
    candidates = sorted(PREFIX.rglob('libdevice*.bc')); assert candidates, 'CUDA libdevice missing'
    libdir = ROOT/'cuda'/'nvvm'/'libdevice'; libdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(candidates[0]), str(libdir/'libdevice.10.bc'))
    command([PREFIX/'bin'/'ptxas', '--version'], SESSION/'environment_check.log', environment(), deadline)
    command([PYTHON, '-c', GPU_CHECK], SESSION/'environment_check.log', environment(), deadline)
    command([PYTHON, '-m', 'pip', 'freeze'], SESSION/'package_versions.txt', environment(), deadline)



assert sys.platform=='linux' and platform.machine()=='x86_64', 'Use Linux x86_64 with an NVIDIA A100'
assert 0<MAX_SESSION_MINUTES<=60
SESSION_STARTED=time.monotonic();DEADLINE=SESSION_STARTED+MAX_SESSION_MINUTES*60
GPU=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total',
    '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
name,memory=GPU.rsplit(',',1)
assert 'A100' in name and float(memory)>=38000, 'Use A100 >= 38,000 MiB; no silent T4/slice/batch fallback.'
print('Timing GPU:',GPU)
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT=Path('/content/drive/MyDrive/sam_replication/igr_a100_one_epoch')
else:
    OUTPUT_ROOT=ROOT/'a100_timing_runs'
SESSION=OUTPUT_ROOT/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
SESSION.mkdir(parents=True,exist_ok=False)
print('TIMING RESULTS:',SESSION,flush=True)
t=time.monotonic()
try:setup(min(DEADLINE,time.monotonic()+1800))
finally:atomic_json(SESSION/'setup_timing.json',dict(seconds=time.monotonic()-t))
ENV=environment()
print('Pinned environment ready. Full experiment will NOT be launched.')
