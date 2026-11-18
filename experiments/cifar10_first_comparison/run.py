#!/usr/bin/env python3
"""Select the two run_config.md experiments using the existing guarded runner.

Default: print a plan, without installing, creating files or using a GPU.
Clone the whole repository: the runner and reviewed notebook are dependencies.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'repro_notebooks' / '04_wrn28_10_cifar10_paper_replication.ipynb'


def validate_config():
    config = json.loads((HERE / 'config.json').read_text())
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    source = next(c for c in cells if 'WORK_ROOT = Path.home()' in c)
    tree = ast.parse(source)
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    actual = {kw.arg: ast.literal_eval(kw.value) for kw in assignments['CONFIG'].keywords
              if kw.arg in ('batch_size', 'logical_replicas', 'subbatch_size', 'lr',
                            'weight_decay', 'rho', 'gradient_clipping', 'budget_limit_hours')}
    for key, value in actual.items():
        if config[key] != value:
            raise ValueError('Notebook/config disagreement: ' + key)
    if config['source_revision'] != ast.literal_eval(assignments['REVISION']):
        raise ValueError('Pinned source revision changed; review the experiment.')
    if config['runs_in_order'] != ['SAM-200_seed_0', 'SGD-200_seed_0']:
        raise ValueError('This entry point is only for the first seed-0 comparison.')
    fixed = dict(dataset='cifar10', model='WideResnet28x10', momentum=0.9, nesterov=True,
                 schedule='cosine, 200 epochs, no warmup',
                 augmentation='basic, no batch-level augmentation')
    fixed_environment = {'python': '3.8', 'pip': '24.0', 'jax': '0.2.6',
                         'jaxlib': '0.1.57+cuda110', 'flax': '0.2.2', 'numpy': '1.18.5',
                         'tensorflow': '2.3.1', 'cudatoolkit': '11.0', 'cudnn': '8.0.5',
                         'cuda-nvcc': '11.3.122', 'runtime_absl_py': '0.10.0'}
    if any(config[key] != value for key, value in fixed.items()) or config['environment'] != fixed_environment:
        raise ValueError('This manifest documents the fixed experiment, not hyperparameter overrides.')
    pins = dict(line.split('==', 1) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if '==' in line)
    for package in ('jax', 'flax', 'numpy', 'tensorflow'):
        if pins[package] != config['environment'][package]:
            raise ValueError('Original dependency pin changed: ' + package)
    for flag in ('RUN_SETUP', 'RUN_PREFLIGHT', 'RUN_TRAINING', 'RUN_DIAGNOSTIC_SMOKE', 'RENDER_RESULTS'):
        if ast.literal_eval(assignments[flag]) is not False:
            raise ValueError('Saved notebook execution switch enabled: ' + flag)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'sam', 'sgd', 'report', 'comparison'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--setup', action='store_true', help='Create/repair the pinned Python 3.8 environment during preflight.')
    parser.add_argument('--session', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ledger-import', type=Path)
    parser.add_argument('--record-l4-smoke', action='store_true')
    parser.add_argument('--deterministic-cudnn', action='store_true', help='Request deterministic cuDNN algorithms in the pinned legacy XLA stack.')
    parser.add_argument('--deterministic-reductions', action='store_true', help='Request deterministic GPU reductions in the pinned legacy XLA stack.')
    parser.add_argument('--check-inputs', action='store_true', help='Read-only filesystem checks before starting tmux.')
    parser.add_argument('--cifar-archive', type=Path)
    parser.add_argument('--max-hours', type=float, default=0.75)
    parser.add_argument('--openstack-bin', default='/workspace/js2-control/bin/openstack')
    parser.add_argument('--leave-active', action='store_true', help='Manual shelving required; avoid for unattended jobs.')
    args = parser.parse_args()
    config = validate_config()
    if not 0 < args.max_hours <= config['budget_limit_hours']:
        parser.error('--max-hours must be positive and at most 40.')
    if args.phase not in ('preflight', 'comparison') and args.session is None:
        parser.error('--session is required after preflight.')
    if args.setup and args.phase not in ('preflight', 'comparison'):
        parser.error('--setup is only valid for preflight/comparison.')
    if args.resume and (args.phase not in ('sam', 'sgd', 'comparison') or args.session is None):
        parser.error('--resume requires a training phase and the original --session.')
    if args.execute or args.check_inputs:
        for name in ('ledger_import', 'cifar_archive'):
            path = getattr(args, name)
            if path is not None and not path.is_file():
                parser.error('--' + name.replace('_', '-') + ' file is missing. Omit this option to reuse persistent inputs.')
        work = Path('/workspace/sam_paper_replication')
        if not (work / 'gpu_budget.json').is_file() and args.ledger_import is None:
            parser.error('Persistent gpu_budget.json missing; recover/import the authoritative ledger first.')
        if not args.setup:
            for path in (work / 'envs/sam/bin/python', work / 'envs/sam/bin/ptxas',
                         work / 'cuda/nvvm/libdevice/libdevice.10.bc', work / 'work/sam/requirements.txt'):
                if not path.is_file():
                    parser.error('Required setup file missing: ' + str(path) + '; use --setup only to install/repair.')
            revision = subprocess.check_output(['git', '-C', str(work / 'work/sam'), 'rev-parse', 'HEAD'], text=True).strip()
            if revision != config['source_revision']:
                parser.error('Runtime source revision differs; do not overwrite it or start training.')
            print('Reusing installed environment and persistent data; no package installation.', flush=True)
    phase = 'train' if args.phase in ('sam', 'sgd') else args.phase
    selection = config['runs_in_order'][1 if args.phase == 'sgd' else 0]
    command = [sys.executable, '-u', str(ROOT / 'experiments' / 'jetstream_wrn.py'), phase,
               '--selection', selection, '--max-hours', str(args.max_hours),
               '--openstack-bin', args.openstack_bin]
    for option in ('session', 'ledger_import', 'cifar_archive'):
        value = getattr(args, option)
        if value is not None:
            command.extend(['--' + option.replace('_', '-'), str(value)])
    for option in ('execute', 'setup', 'resume', 'leave_active', 'record_l4_smoke', 'deterministic_cudnn', 'deterministic_reductions'):
        if getattr(args, option):
            command.append('--' + option.replace('_', '-'))
    print('First comparison: SAM-200 seed 0, then SGD-200 seed 0; no SGD-400 or extra seeds.', flush=True)
    print('Selected:', selection, '| worker budget: 40 cumulative hours | invocation cap:', args.max_hours, flush=True)
    if not args.execute:
        print('PLAN ONLY: no installation, GPU, network, filesystem writes or cloud actions.')
        print('Delegated command:', command)
        return
    if sys.platform != 'linux':
        parser.error('Execute on csci567-sam Linux; local Windows use is plan-only.')
    if args.phase == 'sgd':
        status = args.session / 'train' / config['runs_in_order'][0] / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'complete':
            parser.error('Complete SAM in this session first, as specified in run_config.md.')
    # Replace this process so the existing supervisor, signal handling and auto-shelving
    # remain authoritative; no nested launcher can outlive the wall-time deadline.
    import os
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
