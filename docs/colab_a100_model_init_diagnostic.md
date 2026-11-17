# Colab A100 model-initialization diagnostic

This is a later-use diagnostic for the pinned SAM environment. It is not a
Jetstream2 validation and does not run training. The diagnostic stops unless
Python 3.8, JAX 0.2.6, jaxlib 0.1.57, and Flax 0.2.2 are already installed.

The Colab CLI supports Linux and macOS, not native Windows. Run these commands
from Linux, macOS, or an approved Linux compatibility environment. They may
consume Colab compute units.

## Provision and verify the runtime

```bash
colab new -s sam-a100-init --gpu A100 --high-mem
colab status -s sam-a100-init
```

Confirm that the status reports an A100 before continuing. Do not use a
different accelerator for this diagnostic.

## Create the bounded diagnostic locally

```bash
cat > /tmp/sam_a100_model_init.py <<'PY'
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

assert sys.version_info[:2] == (3, 8), "This diagnostic requires Python 3.8."

import numpy as np
import jax
import jaxlib
import flax

assert jax.__version__ == "0.2.6", jax.__version__
assert jaxlib.__version__.split("+")[0] == "0.1.57", jaxlib.__version__
assert flax.__version__ == "0.2.2", flax.__version__
assert any(device.platform == "gpu" for device in jax.devices()), jax.devices()

repo = Path("/tmp/sam-replication")
subprocess.run(
    ["git", "clone", "--branch", "feat/jetstream-wrn-runner",
     "https://github.com/Siddharth-magesh/sam-replication.git", str(repo)],
    check=True,
)
sys.path.insert(0, str(repo))

from sam_jax.models import load_model, wide_resnet
from flax import serialization

module = wide_resnet.WideResnet.partial(
    blocks_per_group=4, channel_multiplier=10, num_outputs=10
)
model, state = load_model.create_image_model(
    jax.random.PRNGKey(0), 32, 32, module, 3
)

for leaf in jax.tree_leaves(model.params):
    if hasattr(leaf, "block_until_ready"):
        leaf.block_until_ready()
for leaf in jax.tree_leaves(state):
    if hasattr(leaf, "block_until_ready"):
        leaf.block_until_ready()

parameters = sum(int(leaf.size) for leaf in jax.tree_leaves(model.params))
assert parameters > 0
digest = hashlib.sha256(serialization.to_bytes(model.params)).hexdigest()
print("python:", platform.python_version())
print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__, "flax:", flax.__version__)
print("devices:", jax.devices())
print("parameters:", parameters)
print("parameter_sha256:", digest)
print("single WRN model initialization: PASS")
PY
```

## Execute through Colab

```bash
colab exec -s sam-a100-init -f /tmp/sam_a100_model_init.py
```

This intentionally constructs the actual WRN-28-10 model once. It does not
construct a second reference model, run a training step, download CIFAR-10,
modify the repository, or alter the Jetstream ledger.

## Preserve output and release the runtime

```bash
colab log -s sam-a100-init -o sam_a100_model_init.log
colab stop -s sam-a100-init
colab sessions
```

Confirm that `sam-a100-init` is no longer listed. A successful result only
shows that this isolated model initialization works in Colab's pinned
environment; it does not prove Jetstream2 driver, CUDA, or full-training
compatibility.
