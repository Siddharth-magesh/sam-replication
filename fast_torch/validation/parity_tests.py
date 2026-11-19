"""CPU checks and measured discrepancies; cross-JAX update parity remains separate."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import torch
from torch.func import functional_call
from torch.nn import functional as F
from fast_torch.models.wrn28_10 import WideResNet
from fast_torch.optimizers.sam import build_gradient, clip, norm, Stepper
from fast_torch.validation.compare_initialization import difference
from fast_torch.data.augmentations import epoch_plan, augment

def metrics(a, b):
    return difference(torch.cat([v.flatten() for v in a.values()]), torch.cat([b[k].flatten() for k in a]))

def manual_gradient(model, state, x, y, rho):
    """Independent autograd oracle, sequential replicas and explicit L2 at each weight set."""
    base = dict(model.named_parameters()); lanes = len(next(iter(state.values())))
    accum = {k: torch.zeros_like(v) for k, v in base.items()}; saved = {k: [] for k in state}
    for i, (images, labels) in enumerate(zip(x.chunk(lanes), y.chunk(lanes))):
        old = {k: v[i] for k, v in state.items()}
        logits, updated = functional_call(model, base, (images, old, True))
        loss = F.cross_entropy(logits, labels)+.00025*sum(v.square().sum() for v in base.values())
        g = torch.autograd.grad(loss, tuple(base.values()))
        if rho:
            length = torch.sqrt(sum(v.square().sum() for v in g))
            perturbed = {k: (v+rho*d/length).detach().requires_grad_() for (k, v), d in zip(base.items(), g)}
            logits, _ = functional_call(model, perturbed, (images, old, True))
            loss = F.cross_entropy(logits, labels)+.00025*sum(v.square().sum() for v in perturbed.values())
            g = torch.autograd.grad(loss, tuple(perturbed.values()))
        for k, value in zip(accum, g): accum[k] += value/lanes
        for k in updated: saved[k].append(updated[k])
    return accum, {k: torch.stack(v) for k, v in saved.items()}

def run():
    torch.set_num_threads(2); torch.manual_seed(0)
    full = WideResNet()
    count = sum(p.numel() for p in full.parameters())
    assert count == full.expected_parameters()
    del full
    model = WideResNet(blocks=1, width=1, classes=3)
    state = model.initial_bn(lanes=8)
    x = torch.randn(16, 3, 8, 8); y = torch.arange(16)%3
    from fast_torch.models.wrn28_10 import key
    from fast_torch.data.augmentations import normalize
    with torch.no_grad():
        _, updated = model(x[:2], {k: v[0] for k, v in state.items()}, True)
        first = F.conv2d(F.pad(x[:2], (1, 1, 1, 1)), model.p[key('init_conv/kernel')])
        mean = first.mean((0, 2, 3)); var = first.square().mean((0, 2, 3))-mean.square()
        root = 'WideResnetGroup_0/WideResnetBlock_0/init_bn/'
        torch.testing.assert_close(updated[root+'mean'], .1*mean)
        torch.testing.assert_close(updated[root+'var'], .9+.1*var)
        images = torch.arange(3*32*32).remainder(256).to(torch.uint8).reshape(1, 3, 32, 32)
        actual = augment(images, torch.tensor([True]), torch.tensor([0]), torch.tensor([8]))
        expected = normalize(F.pad(images.flip(-1), (4, 4, 4, 4), mode='reflect')[:, :, :32, 8:40])
        assert torch.equal(actual, expected)
    result = {'parameter_count': count, 'cpu_checks': {}, 'jax_update_parity': 'unverified'}
    for rho in (0., .05):
        before = {k: v.clone() for k, v in model.named_parameters()}
        a, sa, _ = build_gradient(model, rho=rho)(dict(model.named_parameters()), state, x, y)
        repeat, sr, _ = build_gradient(model, rho=rho)(dict(model.named_parameters()), state, x, y)
        b, sb, _ = build_gradient(model, rho=rho, strategy='vectorized')(dict(model.named_parameters()), state, x, y)
        oracle, so = manual_gradient(model, state, x, y, rho)
        # Fixed arithmetic check budget, declared before measurement; not a JAX parity tolerance.
        for other in (repeat, b, oracle):
            for k in a: torch.testing.assert_close(a[k], other[k], atol=2e-5, rtol=2e-4)
        for other in (sr, sb, so):
            for k in sa: torch.testing.assert_close(sa[k], other[k], atol=2e-5, rtol=2e-4)
        for k, v in model.named_parameters(): assert torch.equal(v, before[k]), 'Perturbation mutated base weights'
        result['cpu_checks'][str(rho)] = dict(vectorized_gradient=metrics(a, b), oracle_gradient=metrics(a, oracle),
            bn=metrics(sa, sb), repeatability=metrics(a, repeat))
        if rho:
            no_l2, _, _ = build_gradient(model, rho=rho, wd=0.)(dict(model.named_parameters()), state, x, y)
            wrong = {k: no_l2[k]+.0005*v for k, v in model.named_parameters()}
            assert metrics(a, wrong)['max_abs'] > 1e-7, 'Test must distinguish misplaced weight decay'
        assert float(norm(clip({k: v*100 for k, v in a.items()}))) <= 5.00001
        trained = copy.deepcopy(model); stepper = Stepper(trained, rho=rho)
        states, loss = stepper.step(state, x, y, .1)
        assert torch.isfinite(loss)
        for k, v in trained.named_parameters():
            torch.testing.assert_close(v, before[k]-.1*1.9*clip(a)[k], atol=2e-5, rtol=2e-4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'checkpoint.pt'
            torch.save(dict(model=trained.state_dict(), bn=states, optimizer=stepper.optimizer.state_dict()), path)
            saved = torch.load(path, weights_only=True)
            restored = copy.deepcopy(model); restored.load_state_dict(saved['model'])
            restored_stepper = Stepper(restored, rho=rho); restored_stepper.optimizer.load_state_dict(saved['optimizer'])
            s1, _ = stepper.step(states, x, y, .09)
            s2, _ = restored_stepper.step(saved['bn'], x, y, .09)
            assert all(torch.equal(a, b) for a, b in zip(trained.parameters(), restored.parameters()))
            assert all(torch.equal(s1[k], s2[k]) for k in s1)
    plan = epoch_plan(0, 2, 17)
    assert all(torch.equal(a, b) for a, b in zip(plan, epoch_plan(0, 2, 17)))
    # BN initialization/evaluation lane averaging and biased variance are explicit functional state.
    mean_state = {k: v.mean(0) for k, v in state.items()}
    with torch.no_grad():
        output, unchanged = model(x, mean_state, False)
        assert all(torch.equal(unchanged[k], mean_state[k]) for k in unchanged)
        assert torch.isfinite(output).all()
    result['status'] = 'cpu_passed_gpu_and_jax_unverified'
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path)
    args = p.parse_args(); results = run()
    if args.output: args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
