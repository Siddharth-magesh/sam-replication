CFG=dict(epochs=1,schedule_epochs=100,train_per_class=450,batch_size=128,seed=42,split_seed=2026,
    lr=0.1,momentum=0.9,weight_decay=0.0005,clip=5.0,rho=0.05,igr_lambda=0.01,
    warmup_steps=10,timed_steps=20,measure_checkpoint=MEASURE_CHECKPOINT,
    worker_seconds=MAX_SESSION_MINUTES*60,repo=str(REPO),output=str(SESSION),
    data_dir=str(ROOT/'tensorflow_datasets'),source_revision=SOURCE_REVISION,gpu=GPU,
    protocol='cifar100_resnet18_a100_one_epoch_timing_v1')
assert CFG['epochs']==1 and CFG['batch_size']==128 and CFG['train_per_class']==450
compile(WORKER_SOURCE,'<A100 timing worker>','exec')
SCRIPT=SESSION/'timing_worker.py';SCRIPT.write_text(WORKER_SOURCE)
atomic_json(SESSION/'config.json',CFG)
MANIFEST=dict(worker_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    config_sha256=hashlib.sha256((SESSION/'config.json').read_bytes()).hexdigest())
atomic_json(SESSION/'manifest.json',MANIFEST)
DATA_SOURCE=r"""
import os,time,json,sys
from pathlib import Path
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
start=time.monotonic();c=json.loads(Path(sys.argv[1]).read_text())
builder=tfds.builder('cifar100:3.0.2',data_dir=c['data_dir']);builder.download_and_prepare()
assert builder.info.splits['train'].num_examples==50000 and builder.info.splits['test'].num_examples==10000
Path(c['output'],'data_setup.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,
    train_available=50000,validation_held_out=5000,test_loaded=False),indent=2))
print('Dataset cache ready. Official test images were not loaded.',flush=True)
"""
compile(DATA_SOURCE,'<dataset setup>','exec')
data_script=SESSION/'prepare_data.py';data_script.write_text(DATA_SOURCE)
t=time.monotonic()
try:command([PYTHON,data_script,SESSION/'config.json'],SESSION/'data_setup.log',ENV,DEADLINE)
finally:atomic_json(SESSION/'data_host_timing.json',dict(seconds=time.monotonic()-t))
