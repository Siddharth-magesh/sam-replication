"""Create a NEW environment only; target inspection and installation require --execute."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv

def main():
    p = argparse.ArgumentParser(); p.add_argument('--prefix', required=True, type=Path)
    p.add_argument('--cpu', action='store_true'); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute: print('PLAN: inspect target, create NEW isolated environment; no installation.'); return
    root = Path(__file__).resolve().parents[2]
    assert not args.prefix.exists(), 'Refusing to modify any existing environment'
    assert root not in args.prefix.resolve().parents, 'Environment must be outside checkout'
    assert (3, 10) <= sys.version_info[:2] <= (3, 13), 'Use Python 3.10-3.13'
    info = dict(python=sys.version, mode='cpu' if args.cpu else 'cuda', torch='2.9.0')
    index = 'https://download.pytorch.org/whl/cpu'
    if not args.cpu:
        assert sys.platform == 'linux'
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip()
        assert not processes, 'GPU job active; leave running JAX alone'
        target = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
        assert len(target) == 1
        name, driver, memory = map(str.strip, target[0].split(','))
        assert 'A100' in name and float(memory) >= 30000
        # Conservative CUDA 12.6 GA native-driver floor from NVIDIA release notes.
        assert tuple(map(int, driver.split('.'))) >= (560, 28, 3), 'Driver not verified for chosen CUDA wheel'
        index = 'https://download.pytorch.org/whl/cu126'
        info.update(gpu=name, driver=driver, memory_mib=memory)
    venv.EnvBuilder(with_pip=True).create(args.prefix)
    python = args.prefix/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.9.0', '--index-url', index], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', 'numpy==2.2.6'], check=True)
    info['wheel_index'] = index
    (args.prefix/'installation_manifest.json').write_text(json.dumps(info, indent=2))

if __name__ == '__main__': main()
