



# Full A100 study; scientific training functions preserved from tested notebook 08.
import os, sys, json, time, math, hashlib, platform, subprocess, traceback, gc
BEGIN = time.monotonic()
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax
from flax import nn, optim, serialization
C = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(C['output']); PHASE = sys.argv[2]
assert PHASE in ('train', 'evaluate', 'diagnose')
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, C['repo'])
from sam_jax.imagenet_models.resnet import ResNetBlock
B = C['batch_size']; METHODS = ('SGD','SAM','IGR')

def ready(tree):
    for v in jax.tree_leaves(tree):
        if hasattr(v, 'block_until_ready'): v.block_until_ready()
    return tree

def save(name, value):
    path=OUT/name; tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)

def gpu_memory():
    try:
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        name,used,total=raw.rsplit(',',2)
        return dict(name=name.strip(),used_mib=float(used),total_mib=float(total),
                    measurement='nvidia-smi snapshot, not true peak')
    except Exception as exc:
        return dict(error=str(exc))

def check_time():
    if time.monotonic()-BEGIN > C['worker_seconds']:
        raise TimeoutError('Worker time cap reached; partial runs cannot enter the final comparison.')

def finite(tree):
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_leaves(tree)), 'Non-finite value'

def norm2(tree):
    return sum(jnp.sum(x*x) for x in jax.tree_leaves(tree))

def l2(tree): return jnp.sqrt(norm2(tree))

def add(a,b): return jax.tree_multimap(lambda x,y:x+y,a,b)
def sub(a,b): return jax.tree_multimap(lambda x,y:x-y,a,b)
def mul(a,s): return jax.tree_map(lambda x:x*s,a)

assert jax.__version__=='0.2.6' and flax.__version__=='0.2.2'
assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu', 'One NVIDIA GPU required'
# Seeds for splits, augmentation and initialization are explicit and separate.
builder=tfds.builder('cifar100:3.0.2', data_dir=C['data_dir'])
assert builder.info.splits, 'Prepare CIFAR-100 in the preceding data cell; worker never downloads data.'
assert builder.info.splits['train'].num_examples==50000
train=tfds.as_numpy(builder.as_dataset(split='train',batch_size=-1))
images,labels=train['image'],train['label'].astype(np.int32)
rs=np.random.RandomState(C['split_seed']); ti=[]; vi=[]
for cls in range(100):
    ids=rs.permutation(np.flatnonzero(labels==cls))
    vi.extend(ids[:50]); ti.extend(ids[50:50+C['train_per_class']])
ti=np.array(ti,dtype=np.int32); vi=np.array(vi,dtype=np.int32)
assert len(set(ti.tolist()) & set(vi.tolist()))==0
assert len(ti)==C['train_per_class']*100 and len(vi)==5000
np.savez(OUT/'split_indices.npz',train=ti,validation=vi)
STEPS=len(ti)//B
assert STEPS>0

def batches(epoch, seed):
    rng=np.random.RandomState(seed+1000+epoch); order=rng.permutation(ti)
    for off in range(0,STEPS*B,B):
        ids=order[off:off+B]
        padded=np.pad(images[ids],((0,0),(4,4),(4,4),(0,0)))
        xy=rng.randint(0,9,(B,2)); flip=rng.rand(B)<0.5
        x=np.stack([padded[i,u:u+32,v:v+32] for i,(u,v) in enumerate(xy)])
        x[flip]=x[flip,:,::-1,:]
        yield x.astype(np.float32)/255.,labels[ids]

def eval_batches(xx, yy, ids):
    for off in range(0,len(ids),B):
        chunk=ids[off:off+B]; n=len(chunk)
        x=np.zeros((B,32,32,3),np.float32); y=np.zeros(B,np.int32)
        x[:n]=xx[chunk].astype(np.float32)/255.; y[:n]=yy[chunk]
        mask=(np.arange(B)<n).astype(np.float32)
        yield x,y,mask

class BatchNorm(nn.Module):
    """Opt-in numerical candidate; same state/parameter names as Flax 0.2.2.

    Only used by the diagnostic until GPU evidence supports a protocol change.
    Centered second moments avoid cancellation in E[x*x] - E[x]**2.
    """
    def apply(self, x, use_running_average=False, momentum=0.9,
              epsilon=1e-5, dtype=jnp.float32,
              scale_init=nn.initializers.ones):
        x = jnp.asarray(x, jnp.float32)
        shape = (x.shape[-1],)
        mean_state = self.state('mean', shape, jax.nn.initializers.zeros)
        var_state = self.state('var', shape, jax.nn.initializers.ones)
        if use_running_average:
            mean, var = mean_state.value, var_state.value
        else:
            axes = tuple(range(x.ndim-1))
            mean = jnp.mean(x, axis=axes)
            var = jnp.mean(jnp.square(x-mean), axis=axes)
            if not self.is_initializing():
                mean_state.value = momentum*mean_state.value+(1-momentum)*mean
                var_state.value = momentum*var_state.value+(1-momentum)*var
        # Do not hide an already corrupt negative running variance by clipping it.
        scale = self.param('scale', shape, scale_init)
        bias = self.param('bias', shape, jax.nn.initializers.zeros)
        multiplier = jax.lax.rsqrt(var+epsilon)*scale
        return jnp.asarray((x-mean)*multiplier+bias, dtype)


class CifarResNet18(nn.Module):
    def apply(self,x,train=True):
        # Raw [0,1] pixel coordinates for IGR and attacks, normalization INSIDE model.
        x=(x-jnp.array([0.5071,0.4867,0.4408]))/jnp.array([0.2675,0.2565,0.2761])
        conv=nn.Conv.partial(bias=False,dtype=jnp.float32)
        bn_module = BatchNorm if C.get('bn_implementation') == 'centered' else nn.BatchNorm
        bn=bn_module.partial(use_running_average=not train,momentum=0.9,epsilon=1e-5,dtype=jnp.float32)
        x=nn.relu(bn(conv(x,64,(3,3),(1,1),padding='SAME',name='stem'),name='stem_bn'))
        for stage in range(4):
            for block in range(2):
                stride=(2,2) if stage>0 and block==0 else (1,1)
                x=ResNetBlock(x,64*2**stage,conv=conv,norm=bn,act=nn.relu,strides=stride,
                              name='stage%d_block%d'%(stage,block))
        return nn.Dense(jnp.mean(x,axis=(1,2)),100)

def fresh(seed):
    with nn.stateful() as state:
        _,params=CifarResNet18.init_by_shape(jax.random.PRNGKey(seed),[((B,32,32,3),jnp.float32)])
    model=nn.Model(CifarResNet18,params)
    return optim.Momentum(learning_rate=C['lr'],beta=C['momentum'],nesterov=False).create(model),state

def losses(logits,y):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[:,None],axis=1)[:,0]

def predict(model,state,x):
    with nn.stateful(state,mutable=False): return model(x,train=False)

def train_loss(model,state,x,y):
    with nn.stateful(state) as ns: z=model(x,train=True)
    ce=jnp.mean(losses(z,y))
    return ce+0.5*C['weight_decay']*norm2(model.params),(ns,ce)

# With frozen BN there is no cross-example dependence. Gradient of the SUM
# returns each image's CE gradient, without the 1/B scaling of gradient(mean CE).
def input_grads(model,state,x,y):
    return jax.grad(lambda pixels:jnp.sum(losses(predict(model,state,pixels),y)))(x)

def penalty(model,state,x,y):
    dx=input_grads(model,state,x,y)
    return jnp.mean(jnp.sum(dx*dx,axis=(1,2,3)))

def igr_loss(model,state,x,y,lam):
    base,(ns,ce)=train_loss(model,state,x,y)
    # Differentiate THROUGH dx to parameters. Only BN running state is constant.
    p=penalty(model,state,x,y)
    return base+lam*p,(ns,ce,p)

def make_step(method):
    @jax.jit
    def step(opt,state,x,y,lr,lam):
        if method=='IGR':
            (value,(ns,ce,p)),g=jax.value_and_grad(igr_loss,has_aux=True)(opt.target,state,x,y,lam)
        else:
            (value,(ns,ce)),g=jax.value_and_grad(train_loss,has_aux=True)(opt.target,state,x,y)
            p=jnp.float32(0.)
            if method=='SAM':
                delta=mul(g,C['rho']/jnp.maximum(l2(g),1e-12))
                perturbed=add(opt.target,delta)
                (_, _),g=jax.value_and_grad(train_loss,has_aux=True)(perturbed,state,x,y)
                # Update ORIGINAL weights; keep clean first-pass BN statistics only.
        gn=l2(g); g=mul(g,jnp.minimum(1.,C['clip']/jnp.maximum(gn,1e-12)))
        return opt.apply_gradient(g,learning_rate=lr),ns,dict(objective=value,clean_ce=ce,igr_penalty=p,gradient_norm=gn)
    return step
STEPS_FN={m:make_step(m) for m in METHODS}

@jax.jit
def clean_batch(model,state,x,y,mask):
    z=predict(model,state,x)
    return jnp.sum((jnp.argmax(z,axis=1)==y)*mask),jnp.sum(losses(z,y)*mask)

def evaluate(model,state,xx,yy,ids):
    acc=ce=0.
    for x,y,mask in eval_batches(xx,yy,ids):
        check_time(); a,b=ready(clean_batch(model,state,x,y,mask)); finite((a,b))
        acc+=float(a); ce+=float(b)
    return dict(accuracy=100*acc/len(ids),ce=ce/len(ids),n=len(ids))


def validate_igr_math():
    # Independent analytic mixed-derivative check, catches a detached penalty or
    # accidental gradient-of-batch-mean / batch-size scaling.
    x=jnp.array([[.2,.7],[.5,.1]],dtype=jnp.float32); w=jnp.array([.3,-.4],dtype=jnp.float32)
    def toy(ww,xx): return .5*jnp.square(xx@ww)
    def p(ww,xx):
        dx=jax.grad(lambda z:jnp.sum(toy(ww,z)))(xx)
        return jnp.mean(jnp.sum(dx*dx,axis=1))
    expected=jnp.mean(jnp.square(x@w))*jnp.sum(w*w)
    mixed=2*jnp.mean((x@w)[:,None]*x,axis=0)*jnp.sum(w*w)+2*w*jnp.mean(jnp.square(x@w))
    np.testing.assert_allclose(np.asarray(p(w,x)),np.asarray(expected),rtol=1e-5)
    np.testing.assert_allclose(np.asarray(jax.grad(p)(w,x)),np.asarray(mixed),rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(np.asarray(p(w,x)),np.asarray(p(w,jnp.concatenate([x,x]))),rtol=1e-5)
    assert float(ready(l2(jax.grad(p)(w,x))))>0
    return dict(toy_penalty=float(p(w,x)),mixed_gradient=np.asarray(mixed).tolist(),batch_duplication_invariant=True)

@jax.jit
def sensitivity_batch(model,state,x,y,mask):
    dx=input_grads(model,state,x,y); sq=jnp.sum(dx*dx,axis=(1,2,3))
    return jnp.sum(jnp.sqrt(sq)*mask),jnp.sum(sq*mask)

@jax.jit
def pgd_batch(model,state,x,y,mask,key):
    eps=C['pgd_epsilon']; alpha=C['pgd_alpha']
    lo=jnp.maximum(0.,x-eps); hi=jnp.minimum(1.,x+eps)
    adv=jnp.clip(x+jax.random.uniform(key,x.shape,minval=-eps,maxval=eps),lo,hi)
    robust=(jnp.argmax(predict(model,state,x),axis=1)==y)
    def body(_,carry):
        adv,robust=carry
        robust=robust & (jnp.argmax(predict(model,state,adv),axis=1)==y)
        grad=input_grads(model,state,adv,y)
        adv=jnp.clip(adv+alpha*jnp.sign(grad),lo,hi)
        return adv,robust
    adv,robust=jax.lax.fori_loop(0,C['pgd_steps'],body,(adv,robust))
    robust=robust & (jnp.argmax(predict(model,state,adv),axis=1)==y)
    return robust,jnp.max(jnp.abs(adv-x)),jnp.min(adv),jnp.max(adv)

@jax.jit
def weight_batch(model,state,x,y,mask):
    def f(m): return jnp.sum(losses(predict(m,state,x),y)*mask)
    return jax.value_and_grad(f)(model)

@jax.jit
def weight_project(delta,radius):
    return mul(delta,jnp.minimum(1.,radius/jnp.maximum(l2(delta),1e-12)))

@jax.jit
def ascent(delta,g,radius):
    return weight_project(add(delta,mul(g,radius/2./jnp.maximum(l2(g),1e-12))),radius)

def probe_weight(model,state,xx,yy,ids):
    val=0.; grad=jax.tree_map(jnp.zeros_like,model)
    for x,y,mask in eval_batches(xx,yy,ids):
        check_time(); v,g=ready(weight_batch(model,state,x,y,mask)); finite((v,g))
        val+=float(v); grad=add(grad,g)
    return val/len(ids),mul(grad,1./len(ids))

def sharpness(model,state,xx,yy,ids,seed):
    base,_=probe_weight(model,state,xx,yy,ids); result=[]
    leaves,structure=jax.tree_flatten(model)
    for radius in C['sharp_radii']:
        best=base; best_norm=0.; trajectory=[]
        for restart in range(C['sharp_restarts']):
            keys=jax.random.split(jax.random.PRNGKey(seed+800+restart),len(leaves))
            delta=jax.tree_unflatten(structure,[jax.random.normal(k,p.shape,dtype=p.dtype) for k,p in zip(keys,leaves)])
            delta=mul(delta,radius/jnp.maximum(l2(delta),1e-12))
            for step in range(C['sharp_steps']+1):
                perturbed=add(model,delta)
                value,g=probe_weight(perturbed,state,xx,yy,ids)
                n=float(ready(l2(delta))); assert n <= radius+1e-5
                trajectory.append(dict(restart=restart,step=step,ce=value,l2=n))
                if value>best: best=value; best_norm=n
                if step<C['sharp_steps']: delta=ready(ascent(delta,g,radius))
        result.append(dict(radius=radius,base_ce=base,max_ce=best,loss_increase=best-base,
                           best_delta_l2=best_norm,weight_l2=float(ready(l2(model))),trajectory=trajectory))
    return result


def state_hash(model, state):
    digest = hashlib.sha256()
    for value in jax.tree_leaves((model.params, state)):
        a = np.asarray(value)
        digest.update(str((a.shape, str(a.dtype))).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def checkpoint_payload(opt, state, epoch):
    return dict(optimizer=opt, state=state, completed_epoch=epoch)


def restore():
    opt, state = fresh(C['seed'])
    ready((opt, state))
    initial = state_hash(opt.target, state)
    path = Path(C['diagnostic_checkpoint']) if PHASE == 'diagnose' else OUT/'latest.msgpack'
    epoch = 0
    if path.exists():
        restored = serialization.from_bytes(checkpoint_payload(opt, state, 0), path.read_bytes())
        opt, state, epoch = restored['optimizer'], restored['state'], int(restored['completed_epoch'])
        assert 0 <= epoch <= C['epochs']
        finite((opt, state))
    return opt, state, epoch, initial


def metric_report(metrics):
    return {key: dict(finite=bool(np.isfinite(np.asarray(value)).all()),
                      value=str(np.asarray(value))) for key, value in metrics.items()}


def bn_report(state):
    values = []
    def walk(obj, path=''):
        if isinstance(obj, dict):
            for key, value in obj.items(): walk(value, path+'/'+str(key))
        elif path.endswith('/var'):
            a = np.asarray(obj)
            values.append(dict(path=path, minimum=str(a.min()), maximum=str(a.max()),
                               finite=bool(np.isfinite(a).all()), negative=int((a < 0).sum())))
    walk(serialization.to_state_dict(state))
    return values


@jax.jit
def bn_forward(model, state, x, y):
    with nn.stateful(state) as updated:
        logits = model(x, train=True)
    return updated, jnp.sum(jnp.argmax(logits, axis=1) == y), jnp.mean(losses(logits, y))


def audit_bn():
    opt, state, completed, _ = restore()
    original = state_hash(opt.target, state)
    optimizer_hash = hashlib.sha256(serialization.to_bytes(opt)).hexdigest()
    result = dict(completed_epoch=completed, bn_before=bn_report(state),
                  note='Diagnostic only: training data for mode comparison/calibration; validation only for scoring; no weight updates.')
    result['validation_before'] = evaluate(opt.target, state, images, labels, vi)
    save('bn_audit.json', result)
    correct_train = correct_eval = ce_train = ce_eval = 0.
    # Identical training images in both modes. Discard each train-mode state update.
    for i, (x, y) in enumerate(batches(0, C['seed'])):
        if i == 10: break
        check_time()
        unused, a, b = ready(bn_forward(opt.target, state, x, y))
        c, d = ready(clean_batch(opt.target, state, x, y, np.ones(B, np.float32)))
        finite((unused, a, b, c, d))
        correct_train += float(a); ce_train += float(b)*B
        correct_eval += float(c); ce_eval += float(d)
    result['training_subset'] = dict(n=10*B,
        batch_statistics_accuracy=100*correct_train/(10*B),
        running_statistics_accuracy=100*correct_eval/(10*B),
        batch_statistics_ce=ce_train/(10*B), running_statistics_ce=ce_eval/(10*B))
    assert state_hash(opt.target, state) == original, 'Mode comparison mutated original state'
    calibrated = state
    for i, (x, y) in enumerate(batches(0, C['seed'])):
        if i == 100: break
        check_time()
        calibrated, a, b = ready(bn_forward(opt.target, calibrated, x, y))
        finite((calibrated, a, b))
        if i % 25 == 0: print('BN calibration batch', i+1, '/100', flush=True)
    result['calibration_training_batches'] = 100
    result['validation_after'] = evaluate(opt.target, calibrated, images, labels, vi)
    result['bn_after'] = bn_report(calibrated)
    assert state_hash(opt.target, state) == original, 'Original state changed'
    assert hashlib.sha256(serialization.to_bytes(opt)).hexdigest() == optimizer_hash, 'Optimizer changed'
    result.update(status='complete', weights_unchanged=True, original_state_unchanged=True)
    save('bn_audit.json', result)


def diagnose():
    if C.get('diagnostic_mode') == 'bn_audit':
        audit_bn()
        return
    opt, state, completed, _ = restore()
    assert 0 <= C['diagnostic_completed_epoch']-completed <= 1 and completed < C['epochs']
    before = dict(completed_epoch=completed, bn=bn_report(state),
                  state_sha256=state_hash(opt.target, state))
    save('diagnostic_before.json', before)
    before['validation'] = evaluate(opt.target, state, images, labels, vi)
    save('diagnostic_before.json', before)
    for i, (x, y) in enumerate(batches(completed, C['seed'])):
        check_time()
        fraction = (completed*STEPS+i)/(C['epochs']*STEPS)
        lr = np.float32(C['lr']*.5*(1+math.cos(math.pi*fraction)))
        new_opt, new_state, metrics = ready(STEPS_FN[C['method']](
            opt, state, x, y, lr, np.float32(C['igr_lambda'])))
        try:
            finite(metrics); finite((new_opt, new_state))
        except AssertionError:
            save('numerical_failure.json', dict(epoch=completed+1, batch=i+1,
                 metrics=metric_report(metrics), bn_before=bn_report(state), bn_after=bn_report(new_state)))
            raise
        opt, state = new_opt, new_state
        if i % 50 == 0: print('Diagnostic batch', i+1, '/', STEPS, flush=True)
    save('diagnostic_result.json', dict(status='finite_epoch_replay',
         bn_implementation=C.get('bn_implementation', 'legacy'), epoch=completed+1,
         batches=STEPS, validation=evaluate(opt.target, state, images, labels, vi),
         bn=bn_report(state), note='One replay only; does not establish full-run stability or scientific success.'))


def train_full():
    method = C['method']
    if method == 'IGR': save('igr_math_check.json', validate_igr_math())
    opt, state, completed, initial = restore()
    history_path = OUT/'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [row for row in history if row['epoch'] <= completed]
    assert [row['epoch'] for row in history] == list(range(1, completed+1)), 'Checkpoint/history mismatch'
    save('history.json', history)
    if completed == C['epochs']:
        validation = evaluate(opt.target, state, images, labels, vi)
    for epoch in range(completed, C['epochs']):
        tick = time.perf_counter()
        ce_sum = p_sum = 0.
        for i, (x, y) in enumerate(batches(epoch, C['seed'])):
            check_time()
            fraction = (epoch*STEPS+i)/(C['epochs']*STEPS)
            lr = np.float32(C['lr']*.5*(1+math.cos(math.pi*fraction)))
            opt, state, metrics = ready(STEPS_FN[method](opt, state, x, y, lr, np.float32(C['igr_lambda'])))
            try:
                finite(metrics)
            except AssertionError:
                save('numerical_failure.json', dict(epoch=epoch+1, batch=i+1,
                     metrics=metric_report(metrics), bn=bn_report(state)))
                raise
            ce_sum += float(metrics['clean_ce']); p_sum += float(metrics['igr_penalty'])
        train_s = time.perf_counter()-tick
        finite((opt, state))
        tick = time.perf_counter()
        validation = evaluate(opt.target, state, images, labels, vi)
        val_s = time.perf_counter()-tick
        row = dict(epoch=epoch+1, train_seconds=train_s, validation_seconds=val_s,
                   train_clean_ce=ce_sum/STEPS, train_igr_penalty=p_sum/STEPS,
                   validation_accuracy=validation['accuracy'], validation_ce=validation['ce'])
        # History first: if interrupted before checkpoint replacement, truncate and replay.
        history.append(row); save('history.json', history)
        path = OUT/'latest.msgpack'; temporary = OUT/'latest.msgpack.tmp'
        tick = time.perf_counter()
        payload = checkpoint_payload(opt, state, epoch+1)
        temporary.write_bytes(serialization.to_bytes(payload)); temporary.replace(path)
        if epoch == completed:
            restored = serialization.from_bytes(payload, path.read_bytes())
            for old, new in zip(jax.tree_leaves(payload), jax.tree_leaves(restored)):
                np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
            del restored
        row['checkpoint_seconds'] = time.perf_counter()-tick
        save('history.json', history)
        print(C['job'], json.dumps(row), flush=True)
    order = np.random.RandomState(C['seed']+1000).permutation(ti)
    save('train_result.json', dict(status='complete', method=method, seed=C['seed'],
         epochs=C['epochs'], igr_lambda=C['igr_lambda'], validation=validation,
         initial_model_bn_sha256=initial, epoch0_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
         final_model_bn_sha256=state_hash(opt.target, state), steps_per_epoch=STEPS,
         checkpoint_sha256=hashlib.sha256((OUT/'latest.msgpack').read_bytes()).hexdigest()))


def evaluate_final():
    assert C['stage'] == 'final', 'Tuning must never evaluate official test images'
    trained = json.loads((OUT/'train_result.json').read_text())
    assert trained['status'] == 'complete'
    assert hashlib.sha256((OUT/'latest.msgpack').read_bytes()).hexdigest() == trained['checkpoint_sha256']
    opt, state, completed, _ = restore()
    assert completed == C['epochs']
    before = state_hash(opt.target, state)
    test = tfds.as_numpy(builder.as_dataset(split='test', batch_size=-1))
    xx, yy = test['image'], test['label'].astype(np.int32)
    assert len(yy) == 10000
    rng = np.random.RandomState(C['probe_seed'])
    probe = np.concatenate([rng.permutation(np.flatnonzero(yy == cls))[:10] for cls in range(100)])
    np.savez(OUT/'test_probe_indices.npz', indices=probe)
    # Independent evaluation units are saved atomically; a stopped probe resumes without retraining.
    clean_path = OUT/'clean.json'
    if not clean_path.exists(): save('clean.json', evaluate(opt.target, state, xx, yy, np.arange(len(yy))))
    sensitivity_path = OUT/'sensitivity.json'
    if not sensitivity_path.exists():
        a = b = 0.
        for x, y, mask in eval_batches(xx, yy, probe):
            check_time(); u, v = ready(sensitivity_batch(opt.target, state, x, y, mask)); finite((u, v))
            a += float(u); b += float(v)
        save('sensitivity.json', dict(input_gradient_l2=a/len(probe), input_gradient_squared_l2=b/len(probe)))
    pgd_path = OUT/'pgd.json'
    if not pgd_path.exists():
        correct = 0.; max_linf = 0.
        for batch, (x, y, mask) in enumerate(eval_batches(xx, yy, probe)):
            survived = np.ones(B, dtype=bool)
            for restart in range(C['pgd_restarts']):
                check_time()
                key = jax.random.PRNGKey(C['probe_seed']+C['seed']*10000+batch*100+restart)
                robust, budget, low, high = ready(pgd_batch(opt.target, state, x, y, mask, key))
                finite((budget, low, high))
                assert float(budget) <= C['pgd_epsilon']+1e-6
                assert float(low) >= -1e-6 and float(high) <= 1+1e-6
                survived &= np.asarray(robust)
                max_linf = max(max_linf, float(budget))
            correct += float(np.sum(survived*mask))
        clean_probe = evaluate(opt.target, state, xx, yy, probe)
        save('pgd.json', dict(pgd_accuracy=100*correct/len(probe), probe_clean_accuracy=clean_probe['accuracy'],
                             max_observed_linf=max_linf, n=len(probe)))
    sharp_path = OUT/'sharpness.json'
    if not sharp_path.exists():
        save('sharpness.json', sharpness(opt.target, state, xx, yy, probe, C['probe_seed']+C['seed']))
    assert state_hash(opt.target, state) == before == trained['final_model_bn_sha256']
    clean = json.loads(clean_path.read_text()); sens = json.loads(sensitivity_path.read_text())
    pgd = json.loads(pgd_path.read_text()); sharp = json.loads(sharp_path.read_text())
    result = dict(status='complete', method=C['method'], seed=C['seed'], igr_lambda=C['igr_lambda'],
                  clean_test_accuracy=clean['accuracy'], test_ce=clean['ce'], weight_l2=float(ready(l2(opt.target.params))))
    result.update(sens); result.update(pgd)
    result.update({'sharpness_rho_'+str(row['radius']): row['loss_increase'] for row in sharp})
    save('metrics.json', result)
    print('FINAL METRICS', json.dumps(result), flush=True)


try:
    live = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    assert live == C['gpu'] and 'A100' in live
    save(PHASE+'_environment.json', dict(python=platform.python_version(), jax=jax.__version__,
        flax=flax.__version__, numpy=np.__version__, tensorflow=tf.__version__, gpu=live))
    save(PHASE+'_status.json', dict(status='running'))
    if PHASE == 'diagnose': diagnose()
    elif PHASE == 'train': train_full()
    else: evaluate_final()
    save(PHASE+'_status.json', dict(status='complete', seconds=time.monotonic()-BEGIN))
except BaseException as exc:
    save(PHASE+'_status.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc(),
                                   seconds=time.monotonic()-BEGIN))
    raise
