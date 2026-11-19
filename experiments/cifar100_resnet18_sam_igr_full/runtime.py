# Read-only runtime validation and bounded subprocess helper from the tested timing launcher.
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
