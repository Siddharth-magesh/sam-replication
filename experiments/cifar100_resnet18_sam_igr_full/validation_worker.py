#!/usr/bin/env python3
"""Evaluation-only extension: execute definitions from a verified saved study worker."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def load_definitions(source):
    tree = ast.parse(source, feature_version=(3, 8))
    # The saved v3 worker ends with its GPU check and train/evaluate dispatch.
    # Keep every scientific definition unchanged, but never run that dispatch.
    assert isinstance(tree.body[-1], ast.Try), 'Unexpected saved-worker entry point'
    entry = tree.body[-1]
    functions = {n.func.id for n in ast.walk(entry) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {'train_full', 'evaluate_final', 'diagnose'} <= functions
    tree.body.pop()
    return compile(ast.fix_missing_locations(tree), '<verified saved worker definitions>', 'exec')


def main():
    cfg_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(cfg_path.read_text())
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    def save(name, value):
        path = output/name; temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)
    try:
        worker = Path(cfg['saved_worker'])
        assert hashlib.sha256(worker.read_bytes()).hexdigest() == cfg['saved_worker_sha256']
        # restore() in diagnose mode reads the explicit source checkpoint, never output/latest.
        sys.argv = [str(worker), str(cfg_path), 'diagnose']
        ns = dict(__name__='validation_probe_definitions', __file__=str(worker))
        exec(load_definitions(worker.read_text()), ns)
        np, jax = ns['np'], ns['jax']
        source = Path(cfg['source_job'])
        trained = json.loads((source/'train_result.json').read_text())
        assert trained['status']=='complete' and trained['epochs']==cfg['epochs']==100
        checkpoint = source/'latest.msgpack'; evaluation = source/'evaluation_state.msgpack'
        for path, key in ((checkpoint, 'checkpoint_sha256'), (evaluation, 'evaluation_state_sha256')):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==trained[key]
        opt, state, epoch, _ = ns['restore']()
        assert epoch==100 and ns['state_hash'](opt.target, state)==trained['final_model_bn_sha256']
        state = ns['serialization'].from_bytes(state, evaluation.read_bytes())
        ns['finite'](state)
        before = ns['state_hash'](opt.target, state)
        assert before==trained['evaluation_model_bn_sha256']
        optimizer_hash = hashlib.sha256(ns['serialization'].to_bytes(opt)).hexdigest()
        # Each class contributes exactly ten of its fifty held-out validation examples.
        ids = ns['vi'].reshape(100, 50)
        rng = np.random.RandomState(cfg['validation_probe_seed'])
        probe = np.concatenate([rng.permutation(row)[:10] for row in ids])
        assert len(probe)==1000 and len(np.unique(probe))==1000
        assert not set(probe.tolist()).intersection(ns['ti'].tolist())
        assert np.all(np.bincount(ns['labels'][probe], minlength=100)==10)
        np.savez(output/'validation_probe_indices.npz', indices=probe)
        subset_hash = hashlib.sha256(probe.tobytes()).hexdigest()
        clean = ns['evaluate'](opt.target, state, ns['images'], ns['labels'], probe)
        norm = squared = 0.
        for x, y, mask in ns['eval_batches'](ns['images'], ns['labels'], probe):
            ns['check_time']()
            a, b = ns['ready'](ns['sensitivity_batch'](opt.target, state, x, y, mask))
            ns['finite']((a, b)); norm += float(a); squared += float(b)
        assert ns['state_hash'](opt.target, state)==before, 'Evaluation changed model/BN'
        assert hashlib.sha256(ns['serialization'].to_bytes(opt)).hexdigest()==optimizer_hash
        for path, key in ((checkpoint, 'checkpoint_sha256'), (evaluation, 'evaluation_state_sha256')):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==trained[key]
        result = dict(status='complete', job=cfg['job'], seed=cfg['seed'], igr_lambda=cfg['igr_lambda'],
            epochs=epoch, n=1000, validation_probe_sha256=subset_hash,
            validation_accuracy=clean['accuracy'], validation_ce=clean['ce'],
            input_gradient_l2=norm/1000, input_gradient_squared_l2=squared/1000,
            evaluation_model_bn_sha256=before, weights_and_bn_unchanged=True,
            checkpoint_sha256=trained['checkpoint_sha256'],
            evaluation_state_sha256=trained['evaluation_state_sha256'],
            saved_worker_sha256=cfg['saved_worker_sha256'],
            seconds=time.monotonic()-begin,
            environment=dict(python=sys.version.split()[0], jax=jax.__version__,
                             flax=ns['flax'].__version__, numpy=np.__version__),
            note='Post-hoc validation diagnostic, development seed 42; no test access or training updates.')
        save('metrics.json', result); save('status.json', dict(status='complete'))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        save('status.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc(),
                                seconds=time.monotonic()-begin))
        raise


if __name__=='__main__': main()
