"""Load the legacy export; compare tensor/layout identity and forward predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from fast_torch.models.wrn28_10 import WideResNet, key

def load_export(path, model):
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']
    assert metadata['seed'] == 0 and metadata['source_revision'] == 'f7b9ff46ee0deab8004d5508baf68b7e7c3dad1a'
    state = model.initial_bn()
    with np.load(path, allow_pickle=False) as data, torch.no_grad():
        expected = {'param/'+p for p in model.shapes} | {'bn/'+p for p in state} | {'inputs', 'outputs'}
        assert expected <= set(data.files)
        assert all(k in expected or k.startswith(('probe/', 'update/')) for k in data.files), 'Unknown export keys'
        for name, shape in model.shapes.items():
            value = torch.from_numpy(data['param/'+name].copy())
            assert tuple(value.shape) == shape and value.dtype == torch.float32
            model.p[key(name)].copy_(value)
            assert torch.equal(model.p[key(name)].cpu(), value)
        state = {k: torch.from_numpy(data['bn/'+k].copy()).to(next(model.parameters()).device) for k in state}
        inputs = torch.from_numpy(data['inputs'].copy())
        outputs = torch.from_numpy(data['outputs'].copy())
    return state, inputs, outputs, metadata

def difference(a, b):
    a, b = a.detach().double(), b.detach().double()
    return dict(max_abs=float((a-b).abs().max()), relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)))

def main():
    p = argparse.ArgumentParser(); p.add_argument('export', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--updates', action='store_true')
    p.add_argument('--calibration', type=Path, help='Reviewed max_abs/relative_l2 bounds and provenance from measured noise')
    args = p.parse_args()
    model = WideResNet(); state, x, expected, metadata = load_export(args.export, model)
    with torch.no_grad(): a, _ = model(x, state, False); b, _ = model(x, state, False)
    result = dict(status='measured_not_certified', export_sha256=metadata['sha256'],
        parameters_equal=True, parameter_count=sum(p.numel() for p in model.parameters()),
        output_difference=difference(a, expected), repeat_difference=difference(a, b),
        note='Review/calibrate cross-framework tolerances independently before claiming parity.')
    if args.updates:
        import copy
        from fast_torch.optimizers.sam import Stepper
        result['updates'] = {}
        with np.load(args.export, allow_pickle=False) as data:
            x = torch.from_numpy(data['probe/inputs'].copy()); y = torch.from_numpy(data['probe/labels'].copy()).long()
            for method, rho in [('SGD', 0.), ('SAM', .05)]:
                candidate = copy.deepcopy(model); stepper = Stepper(candidate, rho=rho)
                lanes = {k: v.expand(8, -1).clone() for k, v in state.items()}
                new_bn, _ = stepper.step(lanes, x, y, .1)
                expected_params = []; actual_params = []; expected_delta = []; actual_delta = []
                for name in model.shapes:
                    ref = torch.from_numpy(data['update/'+method+'/param/'+name].copy())
                    got = candidate.p[key(name)].detach(); initial = model.p[key(name)].detach()
                    expected_params.append(ref.flatten()); actual_params.append(got.flatten())
                    expected_delta.append((ref-initial).flatten()); actual_delta.append((got-initial).flatten())
                result['updates'][method] = dict(parameters=difference(torch.cat(actual_params), torch.cat(expected_params)),
                    parameter_update=difference(torch.cat(actual_delta), torch.cat(expected_delta)),
                    bn=difference(torch.cat([v.flatten() for v in new_bn.values()]),
                        torch.cat([torch.from_numpy(data['update/'+method+'/bn/'+k].copy()).flatten() for k in new_bn])))
    if args.calibration:
        bounds = json.loads(args.calibration.read_text()); assert bounds['provenance']
        assert args.updates, 'Initialization alone cannot certify SAM/SGD parity'
        comparisons = [result['output_difference'], result['repeat_difference']]
        for item in result['updates'].values(): comparisons.extend(item.values())
        comparisons.extend(metadata['update_repeatability'].values())
        assert set(metadata['update_repeatability']) == {'SAM', 'SGD'}
        result['status'] = 'passed' if all(d['max_abs'] <= bounds['max_abs'] and
            d['relative_l2'] <= bounds['relative_l2'] for d in comparisons) else 'failed'
        result['calibration'] = bounds
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
