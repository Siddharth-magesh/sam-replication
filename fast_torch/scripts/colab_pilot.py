"""Explicit Colab-only two-epoch pilot; leaves CPU/legacy environments untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: isolated CUDA PyTorch 2.9.0, CIFAR download, SAM then SGD, 2 epochs each; 60-minute run cap.')
        return
    assert sys.platform == 'linux' and Path('/content').is_dir(), 'This helper is for Colab'
    assert (3, 10) <= sys.version_info[:2] <= (3, 13)
    assert not args.resume or args.output, 'Resume requires the previous --output directory'
    repo = Path(__file__).resolve().parents[2]
    from fast_torch.benchmarks.benchmark import idle
    idle()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                   '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(gpu) == 1, gpu
    name, driver = map(str.strip, gpu[0].split(','))
    assert 'L4' in name or 'A100' in name, 'This pilot setup targets Colab L4/A100'
    assert tuple(map(int, driver.split('.'))) >= (560, 28, 3), driver
    env = Path('/content/fast_torch_pilot_env')
    marker = env/'pilot_environment.json'
    expected = dict(python=sys.version, torch='2.9.0+cu126', numpy='2.2.6')
    if env.exists():
        assert marker.exists() and json.loads(marker.read_text()) == expected, 'Unrecognized environment; refusing to modify it'
    else:
        env.mkdir()
        marker.write_text(json.dumps(expected, indent=2))
    # without_pip avoids Colab's unavailable ensurepip; host pip targets this venv.
    python = env/'bin/python'
    if not python.exists():
        venv.EnvBuilder(with_pip=False).create(env)
    check = "import torch,numpy; assert torch.__version__=='2.9.0+cu126'; assert numpy.__version__=='2.2.6'"
    ready = subprocess.run([str(python), '-c', check], capture_output=True, text=True)
    if ready.returncode:
        pip = [sys.executable, '-m', 'pip', '--python', str(python), 'install']
        run(pip+['torch==2.9.0', '--index-url', 'https://download.pytorch.org/whl/cu126'])
        run(pip+['numpy==2.2.6'])
    run([str(python), '-c', check])
    # Read the single checksum constant without importing the torch-based data loader.
    import ast
    tree = ast.parse((repo/'fast_torch/data/cifar10.py').read_text())
    checksum = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ARCHIVE_SHA256' for t in n.targets))
    archive = Path('/content/fast_torch_data/cifar-10-binary.tar.gz')
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        print('Downloading CIFAR-10 once; subsequent runs reuse the verified archive.', flush=True)
        partial = archive.with_suffix('.part')
        with urllib.request.urlopen('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', timeout=120) as response:
            with partial.open('wb') as target:
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    target.write(block)
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == checksum, 'Downloaded archive checksum mismatch'
        partial.replace(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum, 'Cached archive checksum mismatch'
    output = args.output or Path('/content/fast_torch_pilot_runs')/str(time.time_ns())
    command = [str(python), '-u', '-B', '-m', 'fast_torch.benchmarks.benchmark',
               '--mode', 'pilot', '--execute', '--minutes', '60', '--methods', 'SAM', 'SGD',
               '--archive', str(archive), '--output', str(output.resolve())]
    if args.resume: command.append('--resume')
    print('Results:', output, '| 2 epochs each; 50-epoch LR schedule; JAX parity UNVERIFIED', flush=True)
    run(command, cwd=repo)


if __name__ == '__main__': main()
