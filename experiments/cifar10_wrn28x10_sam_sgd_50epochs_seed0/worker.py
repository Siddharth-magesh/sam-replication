# Orchestration around the authors' unchanged SAM implementation (Apache-2.0).
import os, sys, json, time, hashlib, traceback, subprocess, platform, shutil
from pathlib import Path
from functools import partial
phase, config_path, label, seed_text, resume_text = sys.argv[1:]
C = json.loads(Path(config_path).read_text())
seed, resume = int(seed_text), resume_text == 'true'
EPOCHS = {'SAM-50':50, 'SGD-50':50}
assert phase in ('preflight','train','smoke') and label in EPOCHS
assert C['mode'] in ('replication','smoke')
assert (phase=='smoke') == (C['mode']=='smoke')
assert phase!='smoke' or (label in ('SAM-50','SGD-50') and seed==0)
# A worker must be started by the persistent budget supervisor.
if 'SAM_STOP_MONOTONIC' not in os.environ:
    raise RuntimeError('Launch through the budget supervisor; direct worker execution is disabled.')
def budget_check():
    if time.monotonic() >= float(os.environ['SAM_STOP_MONOTONIC']):
        raise TimeoutError('Cumulative GPU budget reached; last committed checkpoint is preserved.')
budget_check()
run = Path(C['session'])/phase/(label+'_seed_'+str(seed))
run.mkdir(parents=True, exist_ok=True)
started, prior_seconds = time.monotonic(), 0.0
active_run = False

def write_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def run_experiment():
    global prior_seconds, active_run
    check(os.environ.get('TF_CUDNN_DETERMINISTIC','0')==C['cudnn_deterministic'],
          'cuDNN determinism setting differs from the saved run configuration.')
    print('Recorded TF_CUDNN_DETERMINISTIC='+C['cudnn_deterministic'],flush=True)
    check(os.environ.get('TF_DETERMINISTIC_OPS','0')==C['tf_deterministic_ops'],
          'TF_DETERMINISTIC_OPS differs from saved configuration.')
    reduction_flags=[v for v in os.environ.get('XLA_FLAGS','').split()
                     if v.startswith('--xla_gpu_deterministic_reductions')]
    check(type(C['deterministic_reductions']) is bool, 'Reduction setting must be boolean.')
    check(reduction_flags==['--xla_gpu_deterministic_reductions='+str(C['deterministic_reductions']).lower()],
          'XLA reduction setting differs from saved configuration.')
    runtime_controls=dict(xla_flags=os.environ.get('XLA_FLAGS',''),
        tf_cudnn_deterministic=C['cudnn_deterministic'],tf_deterministic_ops=C['tf_deterministic_ops'])
    print('Recorded XLA_FLAGS='+runtime_controls['xla_flags'],flush=True)
    write_json(run/'runtime_controls.json',runtime_controls)
    import numpy as np
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    import jax, jaxlib, flax
    import jax.numpy as jnp
    from flax import serialization
    from absl import flags
    sys.path.insert(0, str(Path(C['repo']).parent))
    from sam.sam_jax.models import load_model, wide_resnet
    from sam.sam_jax.datasets import dataset_source
    from sam.sam_jax.training_utils import flax_training as official
    FLAGS = flags.FLAGS
    FLAGS(['replication'])
    settings = dict(learning_rate=0.1, weight_decay=0.0005, gradient_clipping=5.0,
        use_rmsprop=False, sam_rho=0.05 if label=='SAM-50' else 0.0,
        sync_perturbations=False, label_smoothing=0.0, no_weight_decay_on_bn=False,
        ema_decay=0.0, use_test_set=True, use_additional_skip_connections=False,
        run_seed=seed, compute_top_5_error_rate=False)
    for key,value in settings.items():
        setattr(FLAGS,key,value)
    versions = dict(python=platform.python_version(), jax=jax.__version__,
        jaxlib=jaxlib.__version__, flax=flax.__version__, tensorflow=tf.__version__, numpy=np.__version__)
    check(sys.version_info[:2]==(3,8), 'Use the pinned Python 3.8 interpreter.')
    check(versions['jax']=='0.2.6' and versions['flax']=='0.2.2', 'JAX/Flax mismatch.')
    check(versions['jaxlib'].split('+')[0]=='0.1.57', 'Expected jaxlib 0.1.57; CUDA execution is checked next.')
    devices = jax.local_devices()
    check(len(devices)==1 and devices[0].platform=='gpu', 'Expose exactly one GPU.')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip()
    def gpu_memory():
        value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
        check(value.isdigit(),'GPU memory query returned a non-numeric value.')
        return int(value)
    # Query the CUDA-visible ordinal, not all GPUs reported by nvidia-smi.
    import ctypes
    cuda = ctypes.CDLL('libcuda.so.1')
    ordinal = ctypes.c_int(); name = ctypes.create_string_buffer(256)
    check(cuda.cuInit(0)==0 and cuda.cuDeviceGet(ctypes.byref(ordinal),0)==0, 'CUDA identity query failed.')
    check(cuda.cuDeviceGetName(name,256,ordinal)==0, 'CUDA device name unavailable.')
    selected_gpu = name.value.decode()
    if C['mode'] in ('replication','smoke'):
        check('A100' in selected_gpu, 'Main replication requires the selected GPU to be an A100.')
    check(versions['tensorflow']=='2.3.1' and versions['numpy']=='1.18.5', 'Legacy TensorFlow/NumPy pins changed.')
    revision = subprocess.check_output(['git','-C',C['repo'],'rev-parse','HEAD'],text=True).strip()
    check(revision==C['revision'], 'Repository revision mismatch.')
    check(not subprocess.check_output(['git','-C',C['repo'],'status','--porcelain','--untracked-files=no'],text=True).strip(), 'Tracked runtime source was modified.')
    source_hashes = {str(p.relative_to(C['repo'])):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(C['repo']).rglob('*.py')) if '.git' not in p.parts}
    check(C['seeds']==[0] and C['batch_size']==256 and C['subbatch_size']==32 and C['logical_replicas']==8
          and C['lr']==0.1 and C['weight_decay']==0.0005 and C['rho']==0.05 and C['gradient_clipping']==5.0,
          'This notebook implements the fixed paper contract; config changes require a reviewed new experiment.')
    identity = dict(config=C, label=label, seed=seed, settings=settings, versions=versions,
        source_hashes=source_hashes, worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if phase == 'train':
        for method in ('SAM-50','SGD-50'):
            pf = Path(C['session'])/'preflight'/(method+'_seed_0')
            report = json.loads((pf/'preflight.json').read_text())
            prior_manifest = json.loads((pf/'manifest.json').read_text())
            check(report['status']=='passed' and json.loads((pf/'status.json').read_text())['status']=='complete', 'Preflight is incomplete.')
            check(report['signature']==prior_manifest['signature'], 'Preflight provenance mismatch.')
            for key in ('config','versions','source_hashes','worker_sha256'):
                check(prior_manifest['identity'][key]==identity[key], 'Environment/source/config changed since preflight: '+key)
    manifest_path = run/'manifest.json'
    if manifest_path.exists():
        check(json.loads(manifest_path.read_text())['signature']==signature, 'Code/config/environment changed: use a new session.')
        old = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        prior_seconds = float(old.get('allocated_wall_seconds',0))
        if old.get('status')=='complete':
            print('Already complete:',run,flush=True)
            return
        check(resume and phase=='train', 'Incomplete existing run: explicitly resume or use a new session.')
    else:
        check(not resume, 'Cannot resume a run without a manifest.')
        write_json(manifest_path,dict(signature=signature,identity=identity))
    active_run = True
    write_json(run/'environment.json',dict(versions=versions,gpu=gpu,selected_gpu=selected_gpu,revision=revision,logical_replicas=8,sam_subbatch=32))
    write_json(run/'status.json',dict(status='running',allocated_wall_seconds=prior_seconds))
    np.random.seed(seed); tf.random.set_seed(seed)
    ds = dataset_source.Cifar10(256,'basic','none')
    check(ds.num_training_obs==50000, 'Expected 50,000 training examples.')
    module = wide_resnet.WideResnet.partial(blocks_per_group=4,channel_multiplier=10,num_outputs=10)
    # Explicit seed bypasses get_model's fixed default and PRNG-array truth test.
    model, initial_state = load_model.create_image_model(jax.random.PRNGKey(seed),32,32,module,3)
    parameter_count = sum(int(x.size) for x in jax.tree_leaves(model.params))
    write_json(run/'architecture_check.json', dict(parameters=parameter_count,
        architecture='WideResnet28x10', matched=True,
        blocks_per_group=4, channel_multiplier=10, num_outputs=10))
    optimizer = official.create_optimizer(model,0.0)
    initialization = dict(seed=seed,
        parameter_sha256=hashlib.sha256(serialization.to_bytes(model.params)).hexdigest(),
        parameters=sum(int(x.size) for x in jax.tree_leaves(model.params)))
    if (run/'initialization.json').exists():
        check(json.loads((run/'initialization.json').read_text()) == initialization,
              'Seeded initialization changed since this run started.')
    else:
        write_json(run/'initialization.json', initialization)
    states = jax.tree_map(lambda x:jnp.stack([x]*8),initial_state)
    schedule = official.get_cosine_schedule(EPOCHS[label],0.1,50000,256)
    original_step = partial(official.train_step,learning_rate_fn=schedule,l2_reg=0.0005)
    # Original pmean averages independent gradients BEFORE clipping and momentum.
    # pmean makes optimizer and LR unmapped. None avoids eight full optimizer copies.
    mapped_step = jax.jit(jax.vmap(original_step,in_axes=(None,0,0,0),
                                  out_axes=(None,0,0,None),axis_name='batch'))
    mapped_eval = jax.jit(jax.vmap(official.eval_step,in_axes=(None,0,0),axis_name='batch'))

    def synchronize(tree):
        for x in jax.tree_leaves(tree):
            if hasattr(x,'block_until_ready'):
                x.block_until_ready()
        return tree

    def finite(tree):
        return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_leaves(tree))

    def keys_for(step,count):
        return jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),int(step)),count)

    def train_batches(epoch):
        # Stateful TF augmentation/GPU kernels need not be bitwise deterministic.
        tf.random.set_seed(seed*100000+epoch)
        options = tf.data.Options(); options.experimental_deterministic=True
        return ds.get_train(True).with_options(options)

    def shard(batch,count=8):
        return {k:np.asarray(v).reshape((count,-1)+tuple(v.shape[1:])) for k,v in batch.items()}

    def differences(a,b):
        check(jax.tree_structure(a)==jax.tree_structure(b), 'Compared state structures differ.')
        max_abs,sqdiff,sqref,first_divergent_leaf = 0.0,0.0,0.0,None
        for leaf_index,(x,y) in enumerate(zip(jax.tree_leaves(a),jax.tree_leaves(b))):
            check(np.shape(x)==np.shape(y), 'Compared array shapes differ.')
            x,y = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
            leaf_max_abs=float(np.max(np.abs(x-y)))
            if leaf_max_abs>0 and first_divergent_leaf is None:
                first_divergent_leaf=leaf_index
            max_abs=max(max_abs,leaf_max_abs)
            sqdiff+=float(np.sum((x-y)**2)); sqref+=float(np.sum(y**2))
        return dict(max_abs=max_abs,relative_l2=(sqdiff**0.5)/max(sqref**0.5,1e-12),
                    first_divergent_leaf=first_divergent_leaf)

    def eval_full(opt,bn,limit_batches=None):
        sums=dict(error_rate=0.0,loss=0.0); seen=0
        for batch_index,batch in enumerate(ds.get_test()):
            if limit_batches is not None and batch_index>=limit_batches:
                break
            budget_check()
            raw={k:np.asarray(v) for k,v in batch.items()}
            n=len(raw['label']); padded=((n+7)//8)*8
            raw={k:np.pad(v,[(0,padded-n)]+[(0,0)]*(v.ndim-1)) for k,v in raw.items()}
            raw['mask']=np.concatenate([np.ones(n),np.zeros(padded-n)]).astype(np.float32)
            out=synchronize(mapped_eval(opt.target,bn,shard(raw)))
            check(finite(out),'Non-finite evaluation.')
            for key in sums:
                sums[key]+=float(np.asarray(out[key])[0])
            seen+=n
        check(seen==(10000 if limit_batches is None else limit_batches*32),'Evaluation sample count mismatch.')
        return dict(test_error_pct=100*sums['error_rate']/seen,
            test_accuracy_pct=100*(1-sums['error_rate']/seen),test_loss=sums['loss']/seen,test_examples=seen)

    history=[]; completed=0
    checkpoints=run/'checkpoints'; checkpoints.mkdir(exist_ok=True)

    def save_checkpoint(epoch,opt,bn,rows):
        folder=checkpoints/('epoch_%04d'%epoch)
        check(not folder.exists(),'Checkpoint exists; refusing overwrite.')
        temporary=checkpoints/('writing_%04d'%epoch); temporary.mkdir(exist_ok=True)
        payload=dict(optimizer=opt,states=bn,completed_epoch=int(epoch))
        with (temporary/'state.msgpack').open('wb') as stream:
            stream.write(serialization.to_bytes(payload)); stream.flush(); os.fsync(stream.fileno())
        write_json(temporary/'history.json',rows)
        write_json(temporary/'metadata.json',dict(signature=signature,completed_epoch=epoch,checkpoint_unit='epoch',
            state_sha256=hashlib.sha256((temporary/'state.msgpack').read_bytes()).hexdigest()))
        os.replace(str(temporary),str(folder))
        write_json(checkpoints/'latest.json',dict(folder=folder.name))
        for old in sorted(checkpoints.glob('epoch_*'))[:-2]:
            check(old.resolve().parent==checkpoints.resolve(),'Unsafe checkpoint cleanup.')
            shutil.rmtree(str(old))
        return folder

    def load_checkpoint():
        committed = sorted(checkpoints.glob('epoch_*'))
        check(bool(committed), 'No complete epoch checkpoint; start a fresh run/session.')
        # Recover even if killed after directory commit but before latest.json update.
        folder=committed[-1]
        check(folder.resolve().parent==checkpoints.resolve(),'Invalid checkpoint pointer.')
        meta=json.loads((folder/'metadata.json').read_text()); data=(folder/'state.msgpack').read_bytes()
        check(meta['signature']==signature and hashlib.sha256(data).hexdigest()==meta['state_sha256'],'Checkpoint provenance mismatch.')
        template=dict(optimizer=optimizer,states=states,completed_epoch=0)
        restored=serialization.from_bytes(template,data)
        rows=json.loads((folder/'history.json').read_text()); epoch=int(restored['completed_epoch'])
        check(epoch==meta['completed_epoch']==len(rows),'Checkpoint/history disagree.')
        check(int(restored['optimizer'].state.step)==epoch*195,'Checkpoint step mismatch.')
        return restored['optimizer'],restored['states'],epoch,rows

    if resume:
        optimizer,states,completed,history=load_checkpoint()
        write_json(run/'history.json',history)
    if phase == 'preflight':
        batch=next(iter(train_batches(0)))
        small={k:np.asarray(v)[:32][None,...] for k,v in batch.items()}
        one_state=jax.tree_map(lambda x:jnp.asarray(x)[None,...],initial_state); one_keys=keys_for(0,1)
        vmapped_one=jax.jit(jax.vmap(original_step,in_axes=(None,0,0,0),out_axes=(None,0,0,None),axis_name='batch'))
        pmapped_one=jax.pmap(original_step,axis_name='batch')
        def clone_arrays(tree):
            return jax.tree_map(lambda x:jnp.array(np.asarray(x)),tree)
        def clone_serialized(value):
            return serialization.from_bytes(value,serialization.to_bytes(value))
        reference_input=(clone_serialized(optimizer),clone_serialized(one_state),clone_arrays(small),clone_arrays(one_keys))
        repeated_input=(clone_serialized(optimizer),clone_serialized(one_state),clone_arrays(small),clone_arrays(one_keys))
        candidate_input=(clone_serialized(optimizer),clone_serialized(one_state),clone_arrays(small),clone_arrays(one_keys))
        input_identity=dict(optimizer_target=differences(reference_input[0].target,repeated_input[0].target),
                            optimizer_state=differences(reference_input[0].state,repeated_input[0].state),
                            batchnorm=differences(reference_input[1],repeated_input[1]),
                            inputs=differences(reference_input[2],repeated_input[2]),
                            rng=differences(reference_input[3],repeated_input[3]))
        check(all(value['max_abs']==0 for value in input_identity.values()),
              'One-replica diagnostic inputs differ before execution.')
        # Prepare independent transformations of the same original train_step.
        def pmap_arguments(inputs):
            # pmap maps axis 0 of EVERY optimizer leaf, including scalar step.
            opt,bn,images,keys=inputs
            replicated=jax.tree_map(lambda x:jnp.asarray(x)[None,...],opt)
            check(all(np.shape(y)==(1,)+np.shape(x) for x,y in
                      zip(jax.tree_leaves(opt),jax.tree_leaves(replicated))),
                  'Reference optimizer must have a leading one-device axis.')
            return replicated,bn,images,keys
        # Compile/warm both transforms before measurement. Reuse the exact same
        # prepared reference arguments, avoiding a separate reconstruction/transfer.
        prepared_reference=synchronize(pmap_arguments(reference_input))
        prepared_candidate=synchronize(candidate_input)
        def input_digest(inputs):
            digest=hashlib.sha256()
            for leaf in jax.tree_leaves(inputs):
                array=np.asarray(leaf)
                digest.update(str((array.dtype.str,array.shape)).encode())
                digest.update(array.tobytes())
            return digest.hexdigest()
        reference_input_hash=input_digest(prepared_reference)
        candidate_input_hash=input_digest(prepared_candidate)
        print('Warming original pmap and candidate vmap before numerical comparison.',flush=True)
        warm=synchronize(pmapped_one(*prepared_reference)); del warm
        warm=synchronize(vmapped_one(*prepared_candidate)); del warm
        reference=synchronize(pmapped_one(*prepared_reference))
        repeated=synchronize(pmapped_one(*prepared_reference))
        candidate=synchronize(vmapped_one(*prepared_candidate))
        input_integrity=dict(reference_unchanged=input_digest(prepared_reference)==reference_input_hash,
                             candidate_unchanged=input_digest(prepared_candidate)==candidate_input_hash)
        write_json(run/'reference_input_integrity.json',input_integrity)
        check(all(input_integrity.values()),'Numerical diagnostic inputs mutated during execution.')
        reference_optimizer=jax.tree_map(lambda x:x[0],reference[0])
        numerical = dict(parameters=differences(candidate[0].target,reference_optimizer.target),
                         optimizer_state=differences(candidate[0].state,reference_optimizer.state),
                         batchnorm=differences(candidate[1],reference[1]),
                         metrics=differences(candidate[2],reference[2]),
                         learning_rate=differences(candidate[3],reference[3][0]))
        checks=dict(one_replica_finite=finite(candidate) and finite(reference),
                    reference_protocol='warmed_same_prepared_inputs_v1',
                    runtime_controls=runtime_controls,
                    input_integrity=input_integrity,
                    input_identity=input_identity,one_replica_difference=numerical)
        write_json(run/'diagnostics.json',checks)
        repeated_optimizer=jax.tree_map(lambda x:x[0],repeated[0])
        repeat_difference=dict(parameters=differences(repeated_optimizer.target,reference_optimizer.target),
                               optimizer_state=differences(repeated_optimizer.state,reference_optimizer.state),
                               batchnorm=differences(repeated[1],reference[1]),
                               metrics=differences(repeated[2],reference[2]),
                               learning_rate=differences(repeated[3],reference[3]))
        strict_close=lambda ds:all(d['max_abs']<=1e-3 and d['relative_l2']<=1e-3 for d in ds.values())
        reference_repeatable=strict_close(repeat_difference)
        cross_transform_close=strict_close(numerical)
        noise_bound=all(numerical[k][metric]<=max(1e-3,3*repeat_difference[k][metric])
                       for k in numerical for metric in ('max_abs','relative_l2'))
        checks.update(reference_repeat_difference=repeat_difference,
                      reference_repeatable=reference_repeatable,
                      cross_transform_close=cross_transform_close,
                      within_reference_noise_bound=noise_bound,
                      numerical_equivalence=('passed' if reference_repeatable and cross_transform_close else
                          'inconclusive_original_not_repeatable' if not reference_repeatable and noise_bound else 'failed'))
        write_json(run/'diagnostics.json',checks)
        print('One-replica numerical comparison:',checks['numerical_equivalence'],flush=True)
        # This is the mandatory strict preflight gate, including original-kernel repeatability.
        acceptable=reference_repeatable and cross_transform_close
        check(checks['one_replica_finite'] and finite(repeated),
              'One-replica diagnostic produced non-finite output.')
        if C.get('diagnostic_only',False):
            write_json(run/'diagnostic_status.json',dict(status='complete',
                reference_repeatable=reference_repeatable,cross_transform_close=cross_transform_close,
                within_reference_noise_bound=noise_bound))
            write_json(run/'status.json',dict(status='complete',diagnostic_only=True,
                allocated_wall_seconds=prior_seconds+time.monotonic()-started))
            print('One-replica diagnostic complete; strict preflight not run.',flush=True)
            return
        check(acceptable,'One-replica numerical verification failed; inspect diagnostics.')
        del repeated,repeated_optimizer,reference,reference_optimizer,candidate,vmapped_one,pmapped_one,reference_input,repeated_input,candidate_input,prepared_reference,prepared_candidate
        # Free discarded comparison buffers before the full eight-lane step.
        import gc
        gc.collect()
        print('Checking eight logical replicas and BatchNorm.',flush=True)
        out=synchronize(mapped_step(optimizer,states,shard(batch),keys_for(0,8)))
        checks['eight_replica_finite']=finite(out)
        checks['shared_optimizer_step']=int(out[0].state.step)==int(optimizer.state.step)+1
        checks['shared_optimizer_shape']=all(np.shape(x)==np.shape(y) for x,y in zip(jax.tree_leaves(out[0]),jax.tree_leaves(optimizer)))
        checks['parameters_changed']=differences(out[0].target,optimizer.target)['max_abs']>0
        # Each lane's retained BN state must equal its CLEAN first pass, including for SAM.
        def clean_bn_forward(model,state,images,key):
            with flax.nn.stateful(state) as updated:
                with flax.nn.stochastic(key):
                    model(images,train=True)
            return updated
        # Match the training transform's lane layout, keeping a separate clean
        # forward with the ORIGINAL weights/state (no optimizer or SAM perturbation).
        # In JAX 0.2.6, vmap with shared convolution weights folds the mapped
        # image axis into the convolution batch; eight serial calls use another shape.
        bn_forward=jax.jit(clean_bn_forward)
        bn_forward_lanes=jax.jit(jax.vmap(clean_bn_forward,
            in_axes=(None,0,0,0),out_axes=0,axis_name='batch'))
        checks['bn_reference_protocol']='clean_forward_same_vmap_layout_v1'
        checks['bn_lanes']=[]
        checks['bn_serial_lanes']=[]  # Preserve the old cross-layout comparison as evidence.
        grouped=shard(batch); lane_keys=keys_for(0,8)
        expected_lanes=synchronize(bn_forward_lanes(optimizer.target,states,grouped['image'],lane_keys))
        checks['bn_reference_finite']=finite(expected_lanes)
        for lane in range(8):
            budget_check()
            state_lane=jax.tree_map(lambda x:x[lane],states)
            serial_expected=synchronize(bn_forward(optimizer.target,state_lane,grouped['image'][lane],lane_keys[lane]))
            expected=jax.tree_map(lambda x:x[lane],expected_lanes)
            actual=jax.tree_map(lambda x:x[lane],out[1])
            checks['bn_lanes'].append(differences(actual,expected))
            checks['bn_serial_lanes'].append(differences(actual,serial_expected))
        checks['bn_updated']=differences(out[1],states)['max_abs']>0
        checks['bn_lane_variation']=max(float(np.max(np.abs(np.asarray(x)-np.asarray(x)[0]))) for x in jax.tree_leaves(out[1]))
        write_json(run/'diagnostics.json',checks)
        check(checks['bn_updated'] and checks['bn_lane_variation']>0,'BN states did not update independently.')
        check(checks['bn_reference_finite'],'Clean BatchNorm reference is non-finite.')
        check(all(d['max_abs']<=2e-5 and d['relative_l2']<=2e-4 for d in checks['bn_lanes']), 'Retained BN differs from clean first pass with matching vmap layout.')
        checks['eight_bn_states']=all(x.shape[0]==8 for x in jax.tree_leaves(out[1]))
        raw_eval=next(iter(ds.get_test()))
        many_eval=synchronize(mapped_eval(optimizer.target,out[1],shard(raw_eval)))
        mean_bn=jax.tree_map(lambda x:jnp.mean(x,axis=0)[None,...],out[1])
        one_eval=jax.jit(jax.vmap(official.eval_step,in_axes=(None,0,0),axis_name='batch'))
        ref_eval=synchronize(one_eval(optimizer.target,mean_bn,shard(raw_eval,1)))
        checks['bn_eval_difference']=differences(jax.tree_map(lambda x:x[0],many_eval),jax.tree_map(lambda x:x[0],ref_eval))
        write_json(run/'diagnostics.json',checks)
        check(checks['eight_replica_finite'] and checks['eight_bn_states'] and checks['shared_optimizer_step'] and checks['shared_optimizer_shape'] and checks['parameters_changed'],
            'Logical-replica update check failed.')
        check(checks['bn_eval_difference']['relative_l2']<=1e-3,'BatchNorm evaluation check failed.')
        del out,many_eval,ref_eval,expected,actual,expected_lanes,serial_expected,bn_forward,bn_forward_lanes
        # Restart timing from untouched initialization; keep the full 50-epoch schedule.
    if phase=='smoke':
        # One complete epoch using the original mapped step; this is functional evidence only.
        initial_parameters=optimizer.target
        epoch_start=time.monotonic(); batch_count=0; losses=[]; memory_samples=[]
        memory_samples.append(dict(step=0,memory_mib=gpu_memory()))
        for batch in train_batches(0):
            budget_check()
            output=synchronize(mapped_step(optimizer,states,shard(batch),keys_for(optimizer.state.step,8)))
            optimizer,states,metrics,lr=output
            check(finite(output),'Non-finite smoke update.')
            batch_count+=1; losses.append(float(np.mean(np.asarray(metrics['train_loss']))))
            if batch_count % 10 == 0:
                memory_samples.append(dict(step=batch_count,memory_mib=gpu_memory()))
        check(batch_count==195,'Smoke epoch batch count mismatch.')
        check(int(optimizer.state.step)==195,'Smoke optimizer step mismatch.')
        check(differences(optimizer.target,initial_parameters)['max_abs']>0,'Smoke training did not update parameters.')
        epoch_seconds=time.monotonic()-epoch_start
        memory_samples.append(dict(step=batch_count,memory_mib=gpu_memory()))
        epoch_row=dict(epoch=1,training_batches=batch_count,mean_loss=float(np.mean(losses)),
                       final_loss=losses[-1],learning_rate=float(np.asarray(lr)),epoch_seconds=epoch_seconds)
        history=[epoch_row]
        save_checkpoint(1,optimizer,states,history)
        ro,rs,re,rh=load_checkpoint()
        check(re==1 and rh==history,'Smoke checkpoint metadata mismatch.')
        check(differences(ro,optimizer)['max_abs']==0 and differences(rs,states)['max_abs']==0,'Smoke checkpoint restoration changed tensors.')
        evaluated=eval_full(optimizer,states)
        restored_eval=eval_full(ro,rs)
        check(abs(evaluated['test_loss']-restored_eval['test_loss'])<=1e-5 and evaluated['test_error_pct']==restored_eval['test_error_pct'], 'Restored evaluation differs.')
        write_json(run/'smoke.json',dict(status='passed',diagnostic_only=True,epochs=1,training_batches=batch_count,
                   training_examples=batch_count*256,evaluation=evaluated,restored_evaluation=restored_eval,
                   checkpoint_restored_at_epochs=[1],parameters_changed=True,finite_losses=True,
                   epoch_runtime_seconds=epoch_seconds,gpu_memory_samples=memory_samples,
                   gpu_memory_peak_mib=max(x['memory_mib'] for x in memory_samples),history=history))
        write_json(run/'status.json',dict(status='complete',diagnostic_only=True,completed_epochs=1,
                   allocated_wall_seconds=prior_seconds+time.monotonic()-started))
        print('One-epoch functional smoke complete; strict preflight not run.',flush=True)
        return
    startup_seconds=time.monotonic()-started
    epoch_limit=1 if phase=='preflight' else EPOCHS[label]
    for epoch in range(completed,epoch_limit):
        budget_check()
        start=time.monotonic(); after_first=None; count=0; step_seconds=[]; tick=start
        for batch in train_batches(epoch):
            budget_check()
            output=synchronize(mapped_step(optimizer,states,shard(batch),keys_for(optimizer.state.step,8)))
            optimizer,states,metrics,lr=output
            check(finite(metrics) and finite(lr),'Non-finite step metrics.')
            now=time.monotonic(); step_seconds.append(now-tick); tick=now
            count+=1
            if count==1 or count%25==0:
                print(label,"epoch",epoch+1,"batch",count,"/195",flush=True)
            if count==1:
                after_first=time.monotonic()
        train_seconds=time.monotonic()-start
        steady_epoch_seconds=(time.monotonic()-after_first)*195/194
        check(count==195,'Expected 195 full batches per epoch.')
        check(finite(optimizer) and finite(states),'Non-finite epoch state.')
        eval_start=time.monotonic(); row=eval_full(optimizer,states)
        row.update(epoch=epoch+1,method=label,seed=seed,last_lr=float(np.asarray(lr)),
            train_seconds=train_seconds,eval_seconds=time.monotonic()-eval_start,
            cumulative_train_seconds=sum(x['train_seconds'] for x in history)+train_seconds)
        history.append(row)
        checkpoint_start=time.monotonic(); save_checkpoint(epoch+1,optimizer,states,history)
        checkpoint_seconds=time.monotonic()-checkpoint_start
        write_json(run/'history.json',history); print(json.dumps(row),flush=True)
        if phase=='preflight':
            ro,rs,re,rh=load_checkpoint()
            check(re==1 and rh==history,'Checkpoint metadata round-trip failed.')
            check(differences(ro,optimizer)['max_abs']==0 and differences(rs,states)['max_abs']==0,'Checkpoint state round-trip failed.')
            write_json(run/'preflight.json',dict(status='passed',signature=signature,checkpoint_roundtrip=True,
                estimated_steady_epoch_seconds=steady_epoch_seconds,evaluation_seconds=row['eval_seconds'],
                checkpoint_seconds=checkpoint_seconds,first_step_and_input_seconds=after_first-start,
                startup_and_diagnostics_seconds=startup_seconds,measured_train_seconds=train_seconds,
                steady_step_mean_seconds=float(np.mean(step_seconds[1:])),
                steady_step_p90_seconds=float(np.percentile(step_seconds[1:],90)),step_seconds=step_seconds,
                planning_epoch_seconds=max(train_seconds,195*float(np.percentile(step_seconds[1:],90)))+row['eval_seconds']+checkpoint_seconds,
                note='Actual synchronized epoch, evaluation and checkpoint timings; p90 steps give conservative planning. Startup includes diagnostic compilation.'))
        write_json(run/'status.json',dict(status='running',completed_epochs=epoch+1,resumed=resume,
            allocated_wall_seconds=prior_seconds+time.monotonic()-started))
    write_json(run/'status.json',dict(status='complete',completed_epochs=epoch_limit,resumed=resume,
        allocated_wall_seconds=prior_seconds+time.monotonic()-started,
        allocated_gpu_hours=(prior_seconds+time.monotonic()-started)/3600))

try:
    run_experiment()
except BaseException as error:
    write_json(run/('failure_%d.json'%int(time.time())),dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    if active_run or not (run/'manifest.json').exists():
        write_json(run/'status.json',dict(status='failed',allocated_wall_seconds=prior_seconds+time.monotonic()-started))
    raise
