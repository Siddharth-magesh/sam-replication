"""Explicit worker: original semantics, isolated outputs, epoch checkpoints, bounded caller."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import torch
from fast_torch.models.wrn28_10 import WideResNet
from fast_torch.optimizers.sam import Stepper
from fast_torch.validation.compare_initialization import load_export
from fast_torch.data.cifar10 import load, train_batches, eval_batches, ARCHIVE_SHA256
from fast_torch.benchmarks.benchmark_configs import CASES, configure

def sync():
    torch.cuda.synchronize()

def atomic(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)

def main(method=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['benchmark', 'train', 'pilot'], required=True)
    p.add_argument('--method', choices=['SAM', 'SGD'], default=method or 'SAM')
    p.add_argument('--case', choices=list(CASES), default='reference')
    p.add_argument('--archive', type=Path, required=True); p.add_argument('--init', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100); p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--resume', action='store_true'); p.add_argument('--approval', type=Path)
    args = p.parse_args()
    assert os.environ.get('FAST_TORCH_SUPERVISED') == '1', 'Use bounded launcher; do not start engine directly'
    assert not args.resume or args.mode in ('train', 'pilot')
    pilot = args.mode == 'pilot'
    assert pilot or args.init, 'Benchmark/full training requires exported JAX initialization'
    assert not pilot or (args.case == 'reference' and args.init is None), 'Pilot uses labeled PyTorch initialization only'
    assert 1 <= args.steps <= 100 and 1 <= args.warmup <= 10
    started = time.perf_counter(); case = CASES[args.case]; configure(case)
    cfg = json.loads(args.config.read_text())
    assert cfg['epochs'] in (50, 200) and cfg['seed'] == 0 and cfg['batch_size'] == 256
    assert cfg['logical_replicas'] == 8 and cfg['lr'] == .1 and cfg['weight_decay'] == .0005
    assert cfg['rho'] == .05 and cfg['momentum'] == .9 and cfg['clip'] == 5.
    assert cfg['subbatch_size'] == 32 and cfg['nesterov'] is True and cfg['warmup'] == 0
    assert cfg['schedule'] == 'cosine' and cfg['augmentation'] == 'basic_reflect'
    assert not pilot or cfg['epochs'] == 50, 'Pilot keeps the 50-epoch cosine schedule'
    assert torch.cuda.is_available(), 'Install CUDA-enabled PyTorch'
    assert pilot or 'A100' in torch.cuda.get_device_name(0), 'A100 validation/benchmark required'
    source = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256(b''.join(str(f.relative_to(source)).encode()+f.read_bytes() for f in sorted(source.rglob('*.py')))).hexdigest()
    identity = dict(config=cfg, method=args.method, case=case, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=platform.python_version(),
        init_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest() if args.init else None, data_sha256=ARCHIVE_SHA256,
        source_sha256=source_hash, gpu=torch.cuda.get_device_name(0),
        cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        independent_implementation=True, jax_update_parity='unverified')
    if pilot:
        identity.update(mode='two_epoch_functional_pilot', stop_after_epochs=2,
                        initialization='PyTorch seed 0, NOT imported JAX weights')
    if args.mode == 'train':
        assert args.approval, 'Training requires reviewed parity/benchmark approval; see README'
        approval = json.loads(args.approval.read_text())
        assert approval['approved'] is True and approval['case'] == args.case
        assert approval['source_sha256'] == source_hash and approval['init_sha256'] == identity['init_sha256']
        assert approval['jax_parity_verified'] is True
        if args.case != 'reference': assert approval['measured_faster_and_parity_passed'] is True
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'manifest.json'
    if manifest.exists():
        assert args.resume and json.loads(manifest.read_text()) == identity, 'Existing output/provenance mismatch'
    else:
        # A resumed comparison may reach its second method for the first time.
        assert not (args.output/'checkpoint.pt').exists(), 'Checkpoint without provenance'
        atomic(manifest, identity)
    torch.manual_seed(cfg['seed'])
    model = WideResNet()
    if args.init:
        state, _, _, _ = load_export(args.init, model)
    else:
        state = model.initial_bn()
        initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        atomic(args.output/'initialization.json', dict(parameters_sha256=initial_hash, seed=cfg['seed']))
    model = model.cuda()
    if case['channels_last']: model.to(memory_format=torch.channels_last)
    state = {k: v.cuda().expand(8, -1).clone() for k, v in state.items()}
    stepper = Stepper(model, rho=.05 if args.method == 'SAM' else 0., strategy=case['strategy'],
        compile=case['compile'], optimizer=case['optimizer'], bf16=case['profile'] == 'bf16_optimized')
    data = load(args.archive, 'cuda' if case['gpu_data'] else 'cpu')
    history = []; first_epoch = 0; global_step = 0
    if args.resume and (args.output/'checkpoint.pt').exists():
        saved = torch.load(args.output/'checkpoint.pt', map_location='cuda', weights_only=True)
        model.load_state_dict(saved['model']); stepper.optimizer.load_state_dict(saved['optimizer'])
        state = saved['bn']; first_epoch = saved['epoch']; global_step = saved['step']; history = saved['history']
    def update(batch, lr):
        nonlocal state
        images, labels = batch
        if case['channels_last']: images = images.contiguous(memory_format=torch.channels_last)
        state, loss = stepper.step(state, images, labels, lr)
        return float(loss)
    if args.mode == 'benchmark':
        batches = iter(train_batches(data['train'], 0, 0, 'cuda'))
        sync(); t = time.perf_counter(); update(next(batches), .1); sync()
        first = time.perf_counter()-t
        t = time.perf_counter()
        for _ in range(args.warmup): update(next(batches), .1)
        sync(); warmup = time.perf_counter()-t
        torch.cuda.reset_peak_memory_stats(); times = []
        for _ in range(args.steps):
            sync(); t = time.perf_counter(); update(next(batches), .1); sync(); times.append(time.perf_counter()-t)
        mean = sum(times)/len(times)
        result = dict(status='measured_candidate_not_validated', case=args.case, method=args.method,
            first_step_seconds=first, compile_inclusive_first_step_seconds=first if case['compile'] else None,
            compile_time_note='First-call wall time includes execution; compiler-only duration not isolated.',
            warmup_seconds=warmup, steady_ms_per_step=1000*mean, images_per_second=256/mean,
            estimated_training_epoch_seconds=195*mean, allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            step_seconds=times, worker_seconds=time.perf_counter()-started, identity=identity)
        if case['compile']:
            from torch._dynamo.utils import compilation_time_metrics
            result['compiler_instrumentation_seconds'] = {k: list(v) for k, v in compilation_time_metrics.items()}
            result['compiler_instrumentation_note'] = 'Nested compiler timings overlap; do not sum them as wall time.'
        atomic(args.output/'benchmark.json', result); print(json.dumps(result), flush=True)
        return
    for epoch in range(first_epoch, 2 if pilot else cfg['epochs']):
        torch.cuda.reset_peak_memory_stats()
        sync(); start = time.perf_counter(); count = 0; train_loss = 0.
        for batch in train_batches(data['train'], cfg['seed'], epoch, 'cuda'):
            lr = .1*.5*(1+math.cos(math.pi*global_step/(195*cfg['epochs'])))
            loss = update(batch, lr); global_step += 1; count += 1; train_loss += loss
            if count == 1 or count%25 == 0: print(args.method, epoch+1, count, '/195 loss', loss, flush=True)
        assert count == 195
        sync(); train_seconds = time.perf_counter()-start
        eval_start = time.perf_counter()
        means = {k: v.mean(0) for k, v in state.items()}; correct = 0; seen = 0; test_loss = 0.
        with torch.no_grad():
            for images, labels in eval_batches(data['test'], 'cuda'):
                logits, _ = model(images, means, False)
                assert bool(torch.isfinite(logits).all()), 'Non-finite evaluation logits'
                test_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
                correct += int((logits.argmax(1) == labels).sum()); seen += len(labels)
        assert seen == 10000
        sync()
        row = dict(epoch=epoch+1, method=args.method, test_error_pct=100*(1-correct/seen),
                   test_accuracy_pct=100*correct/seen, test_loss=test_loss/seen, test_examples=seen,
                   train_seconds=train_seconds, eval_seconds=time.perf_counter()-eval_start, last_lr=lr,
                   train_loss=train_loss/count, train_loss_note='SAM perturbed-pass CE; SGD unperturbed CE',
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        history.append(row)
        temp = args.output/'checkpoint.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=stepper.optimizer.state_dict(), bn=state,
                        epoch=epoch+1, step=global_step, history=history), temp)
        temp.replace(args.output/'checkpoint.pt'); atomic(args.output/'history.json', history)
        print(json.dumps(row), flush=True)
    atomic(args.output/'final.json', dict(status='pilot_complete_parity_unverified' if pilot else 'complete',
                                        final_epoch=history[-1], identity=identity))

if __name__ == '__main__': main()
