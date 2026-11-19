



# Intermediate derived from notebook 06 and notebook 02's JAX/Flax CIFAR ResNet pilot. See notebook protocol.
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

class CifarResNet18(nn.Module):
    def apply(self,x,train=True):
        # Raw [0,1] pixel coordinates for IGR and attacks, normalization INSIDE model.
        x=(x-jnp.array([0.5071,0.4867,0.4408]))/jnp.array([0.2675,0.2565,0.2761])
        conv=nn.Conv.partial(bias=False,dtype=jnp.float32)
        bn=nn.BatchNorm.partial(use_running_average=not train,momentum=0.9,epsilon=1e-5,dtype=jnp.float32)
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

METHOD=PHASE
assert METHOD in METHODS
assert C['epochs']==1 and C['train_per_class']==450 and B==128
assert C['schedule_epochs']==100

def state_hash(model,state):
    digest=hashlib.sha256()
    for v in jax.tree_leaves((model.params,state)):
        a=np.asarray(v)
        digest.update(str((a.shape,str(a.dtype))).encode()); digest.update(a.tobytes())
    return digest.hexdigest()

def step_statistics(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),p10=float(np.percentile(a,10)),
                p90=float(np.percentile(a,90)),maximum=float(np.max(a)),first=float(a[0]))

def run_one_epoch(method):
    # Warm-up and short timing use a sacrificial model. All its state is discarded.
    t=time.perf_counter();opt,state=fresh(C['seed']);ready((opt,state))
    initial_hash=state_hash(opt.target,state); init_s=time.perf_counter()-t
    checks=validate_igr_math() if method=='IGR' else {}
    initial_model=opt.target
    warm=[]; probe=[]; it=batches(0,C['seed'])
    t=time.perf_counter()
    for _ in range(C['warmup_steps']):
        check_time(); tick=time.perf_counter();x,y=next(it)
        opt,state,metrics=ready(STEPS_FN[method](opt,state,x,y,np.float32(C['lr']),np.float32(C['igr_lambda'])))
        finite(metrics);warm.append(time.perf_counter()-tick)
    warm_s=time.perf_counter()-t
    change=float(ready(l2(sub(opt.target,initial_model))))
    assert change>0 and np.isfinite(change)
    if method=='IGR':assert float(metrics['igr_penalty'])>0
    for _ in range(C['timed_steps']):
        check_time();tick=time.perf_counter();x,y=next(it)
        opt,state,metrics=ready(STEPS_FN[method](opt,state,x,y,np.float32(C['lr']),np.float32(C['igr_lambda'])))
        finite(metrics);probe.append(time.perf_counter()-tick)
    stats=step_statistics(probe)
    stable=(stats['maximum']<=3*max(stats['median'],1e-12))
    warm_memory=gpu_memory()
    save(method+'_warmup.json',dict(warmup_seconds=warm_s,warmup_step_seconds=warm,
        timed_step_seconds=probe,timed_step_statistics_seconds=stats,timed_probe_stable=stable,
        parameter_change_l2=change,igr_checks=checks,gpu_memory_snapshot=warm_memory))
    del opt,state,initial_model;gc.collect()

    # Fresh seeded initialization and fresh epoch generator. Warm-up contributes NO updates.
    tick=time.perf_counter();opt,state=fresh(C['seed']);ready((opt,state))
    assert state_hash(opt.target,state)==initial_hash, 'Reset did not reproduce initial model/BN'
    reset_s=time.perf_counter()-tick
    order=np.random.RandomState(C['seed']+1000).permutation(ti)
    order_hash=hashlib.sha256(order.tobytes()).hexdigest()
    it=batches(0,C['seed']); step_times=[];prep_times=[];compute_times=[];trace=[]
    ce_sum=penalty_sum=0.;epoch_start=time.perf_counter()
    for i in range(STEPS):
        check_time(); tick=time.perf_counter();x,y=next(it);prepared=time.perf_counter()
        # Epoch 1 of the proposed 100-epoch cosine schedule, not a one-epoch decay.
        lr=np.float32(C['lr']*.5*(1+math.cos(math.pi*i/(C['schedule_epochs']*STEPS))))
        opt,state,metrics=ready(STEPS_FN[method](opt,state,x,y,lr,np.float32(C['igr_lambda'])))
        end=time.perf_counter();finite(metrics)
        prep_times.append(prepared-tick);compute_times.append(end-prepared);step_times.append(end-tick)
        ce_sum+=float(metrics['clean_ce']);penalty_sum+=float(metrics['igr_penalty'])
        if i<5:trace.append(dict(step=i+1,lr=float(lr),seconds=step_times[-1]))
        if (i+1)%50==0:
            print(method,'epoch 1:',i+1,'/',STEPS,'updates',flush=True)
            save(method+'_progress.json',dict(completed_steps=i+1,total_steps=STEPS,epoch=1))
    epoch_s=time.perf_counter()-epoch_start
    assert len(step_times)==STEPS==351
    try:next(it)
    except StopIteration:pass
    else:raise AssertionError('Expected exactly one epoch')
    train_memory=gpu_memory()
    save(method+'_steps.json',dict(step_seconds=step_times,input_preparation_seconds=prep_times,
        synchronized_update_seconds=compute_times,first_steps=trace,epoch_training_seconds=epoch_s))

    # Compile validation outside its measured window. Only validation, never test, is scored.
    x,y,mask=next(eval_batches(images,labels,vi));tick=time.perf_counter()
    ready(clean_batch(opt.target,state,x,y,mask));eval_compile_s=time.perf_counter()-tick
    tick=time.perf_counter();validation=evaluate(opt.target,state,images,labels,vi);val_s=time.perf_counter()-tick
    checkpoint_s=restore_s=0.;checkpoint_bytes=0
    if C['measure_checkpoint']:
        payload=dict(optimizer=opt,state=state,completed_epoch=1)
        dest=OUT/(method+'_epoch1.msgpack');tmp=dest.with_name(dest.name+'.tmp')
        tick=time.perf_counter();tmp.write_bytes(serialization.to_bytes(payload));tmp.replace(dest)
        checkpoint_s=time.perf_counter()-tick;checkpoint_bytes=dest.stat().st_size
        tick=time.perf_counter();restored=serialization.from_bytes(payload,dest.read_bytes())
        for old,new in zip(jax.tree_leaves(payload),jax.tree_leaves(restored)):
            np.testing.assert_array_equal(np.asarray(old),np.asarray(new))
        restore_s=time.perf_counter()-tick;del restored,payload
    wall=time.monotonic()-BEGIN
    # Includes imports/data load/init/warm-up/probe/reset/eval compilation and optional restore,
    # once per fresh process. Projection deliberately retains this cost conservatively.
    overhead=max(0.,wall-epoch_s-val_s-checkpoint_s)
    result=dict(method=method,epochs_measured=1,schedule_epochs=C['schedule_epochs'],seed=C['seed'],
        igr_lambda=C['igr_lambda'] if method=='IGR' else 0.,batch_size=B,train_examples=len(ti),
        validation_examples=len(vi),steps=STEPS,examples_used=STEPS*B,dropped_examples=len(ti)-STEPS*B,
        parameter_count=sum(v.size for v in jax.tree_leaves(opt.target.params)),
        initial_model_bn_sha256=initial_hash,epoch_order_sha256=order_hash,
        epoch_training_seconds=epoch_s,step_statistics_seconds=step_statistics(step_times),
        summed_input_preparation_seconds=sum(prep_times),summed_synchronized_update_seconds=sum(compute_times),
        initialization_seconds=init_s,warmup_seconds=warm_s,reset_seconds=reset_s,
        warmup_probe_stable=stable,validation_compile_seconds=eval_compile_s,
        validation_seconds=val_s,validation_accuracy_diagnostic_only=validation['accuracy'],
        validation_ce_diagnostic_only=validation['ce'],train_clean_ce=ce_sum/STEPS,
        train_igr_penalty=penalty_sum/STEPS,checkpoint_seconds=checkpoint_s,
        checkpoint_bytes=checkpoint_bytes,checkpoint_restore_check_seconds=restore_s,
        fixed_worker_overhead_seconds=overhead,worker_wall_seconds=wall,
        gpu_memory_after_warmup=warm_memory,gpu_memory_after_epoch=train_memory,
        memory_measurement='nvidia-smi snapshots only; not measured peak memory',
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_revision=C['source_revision'],gpu=C['gpu'],
        environment=dict(python=platform.python_version(),jax=jax.__version__,flax=flax.__version__,
                         numpy=np.__version__,tensorflow=tf.__version__),status='complete')
    save(method+'_result.json',result)
    print('MEASURED ONE EPOCH',json.dumps(result,indent=2),flush=True)
    return result

try:
    save(METHOD+'_status.json',dict(status='running'))
    assert jax.__version__=='0.2.6' and flax.__version__=='0.2.2'
    live_gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    assert live_gpu==C['gpu'] and 'A100' in live_gpu, 'GPU changed since setup'
    run_one_epoch(METHOD)
    save(METHOD+'_status.json',dict(status='complete',wall_seconds=time.monotonic()-BEGIN))
except BaseException as exc:
    save(METHOD+'_status.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc(),
        wall_seconds=time.monotonic()-BEGIN,gpu_memory_snapshot=gpu_memory()))
    raise

