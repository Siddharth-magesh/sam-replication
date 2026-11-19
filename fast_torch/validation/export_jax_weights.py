"""Legacy Python 3.8/JAX only. NOT executed during implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', action='store_true', help='Also export original eight-lane SGD/SAM updates and repeatability')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert repo not in args.output.resolve().parents, 'Export outside checkout'
    assert not args.output.exists() and not args.output.with_suffix('.json').exists()
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    import jax
    from absl import flags
    from sam.sam_jax.models import load_model, wide_resnet
    from sam.sam_jax.training_utils import flax_training as official
    from flax import serialization
    flags.FLAGS(['export']); flags.FLAGS.use_additional_skip_connections = False
    source = Path(load_model.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    assert revision == 'f7b9ff46ee0deab8004d5508baf68b7e7c3dad1a'
    module = wide_resnet.WideResnet.partial(blocks_per_group=4, channel_multiplier=10, num_outputs=10)
    model, state = load_model.create_image_model(jax.random.PRNGKey(args.seed), 32, 32, module, 3)
    arrays = {}
    def flatten(value, prefix=''):
        for k, v in value.items():
            name = (prefix+'/'+str(k)).strip('/')
            if isinstance(v, dict): yield from flatten(v, name)
            else: yield name, np.asarray(v)
    for name, value in flatten(serialization.to_state_dict(model.params)):
        if value.ndim == 4: value = value.transpose(3, 2, 0, 1)
        elif value.ndim == 2: value = value.T
        arrays['param/'+name] = value
    # Collection serialization may wrap module paths; require explicit recognizable BN suffixes.
    raw_state = list(flatten(serialization.to_state_dict(state)))
    for param in list(arrays):
        if not param.endswith('/scale'): continue
        bn = param[len('param/'):-len('/scale')]
        for statistic in ('mean', 'var'):
            suffix = bn+'/'+statistic
            matches = [v for name, v in raw_state if name == suffix or name.endswith('/'+suffix)]
            assert len(matches) == 1, ('Ambiguous/unrecognized BN serialization; inspect legacy schema', suffix, [n for n, _ in raw_state])
            arrays['bn/'+suffix] = matches[0]
    rng = np.random.RandomState(123)
    inputs = rng.standard_normal((8, 32, 32, 3)).astype(np.float32)
    import flax
    with flax.nn.stateful(state, mutable=False): outputs = model(inputs, train=False)
    arrays['inputs'] = inputs.transpose(0, 3, 1, 2); arrays['outputs'] = np.asarray(outputs)
    repeats = {}
    if args.updates:
        from functools import partial
        import jax.numpy as jnp
        settings = dict(learning_rate=.1, weight_decay=.0005, gradient_clipping=5., use_rmsprop=False,
            sync_perturbations=False, label_smoothing=0., no_weight_decay_on_bn=False, ema_decay=0.,
            use_test_set=True, run_seed=args.seed, compute_top_5_error_rate=False)
        for k, v in settings.items(): setattr(flags.FLAGS, k, v)
        images = rng.standard_normal((256, 32, 32, 3)).astype(np.float32)
        labels = np.arange(256) % 10
        arrays['probe/inputs'] = images.transpose(0, 3, 1, 2); arrays['probe/labels'] = labels
        batch = dict(image=images.reshape(8, 32, 32, 32, 3), label=np.eye(10, dtype=np.float32)[labels].reshape(8, 32, 10))
        optimizer = official.create_optimizer(model, 0.)
        states = jax.tree_map(lambda x: jnp.stack([x]*8), state)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), 8)
        def canonical_bn(value):
            raw = list(flatten(serialization.to_state_dict(value))); answer = {}
            for k in arrays:
                if not k.startswith('bn/'): continue
                name = k[3:]; values = [v for path, v in raw if path == name or path.endswith('/'+name)]
                assert len(values) == 1; answer[name] = values[0]
            return answer
        def exported(output):
            opt, bn, _, _ = output
            result = {}
            for name, v in flatten(serialization.to_state_dict(opt.target.params)):
                if v.ndim == 4: v = v.transpose(3, 2, 0, 1)
                elif v.ndim == 2: v = v.T
                result['param/'+name] = v
            result.update({'bn/'+k: v for k, v in canonical_bn(bn).items()})
            return result
        for method, rho in [('SGD', 0.), ('SAM', .05)]:
            flags.FLAGS.sam_rho = rho
            mapped = jax.jit(jax.vmap(partial(official.train_step,
                learning_rate_fn=official.get_cosine_schedule(50, .1, 50000, 256), l2_reg=.0005),
                in_axes=(None, 0, 0, 0), out_axes=(None, 0, 0, None), axis_name='batch'))
            exported(mapped(optimizer, states, batch, keys))  # compile/warm from same unchanged inputs
            first = exported(mapped(optimizer, states, batch, keys))
            second = exported(mapped(optimizer, states, batch, keys))
            delta2 = sum(float(np.square(first[k].astype(np.float64)-second[k]).sum()) for k in first)
            norm2 = sum(float(np.square(first[k].astype(np.float64)).sum()) for k in first)
            repeats[method] = dict(max_abs=max(float(np.abs(first[k]-second[k]).max()) for k in first),
                                   relative_l2=float(np.sqrt(delta2/max(norm2, 1e-60))))
            arrays.update({'update/'+method+'/'+k: v for k, v in first.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f: np.savez(f, **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(dict(seed=args.seed, source_revision=revision,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), jax=jax.__version__,
        parameter_count=sum(v.size for k, v in arrays.items() if k.startswith('param/')),
        update_repeatability=repeats, runtime_controls={k: __import__('os').environ.get(k) for k in
            ['XLA_FLAGS', 'TF_CUDNN_DETERMINISTIC', 'TF_DETERMINISTIC_OPS']}), indent=2))

if __name__ == '__main__': main()
