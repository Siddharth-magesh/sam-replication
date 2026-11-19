# Source audit and implementation decisions

This is independent code, not an authors' implementation or a replacement for existing runs. Audited November 18, 2026. The four local files `models/wide_resnet.py`, `models/utils.py`, `training_utils/flax_training.py` and `datasets/augmentation.py` matched current [authors' source](https://github.com/google-research/sam) byte-for-byte after newline normalization. The runtime/export pin remains `f7b9ff46ee0deab8004d5508baf68b7e7c3dad1a`.

| Local source | Decision |
|---|---|
| `sam_jax/models/wide_resnet.py:72,114,224` | 16-channel initial convolution, three four-block groups of 160/320/640 channels, preactivation, pooled and trailing-zero-padded shortcuts; no learned shortcut convolutions. SAME stride-two padding is asymmetric. |
| `sam_jax/models/utils.py:28,60,64` | BN old-value decay .9, epsilon 1e-5; normal fan-out convolution initialization; classifier uniform ±1/sqrt(10), zero bias. |
| `sam_jax/training_utils/flax_training.py:198,377` | Nesterov .9; 195 steps/epoch, cosine over the selected full duration, no warmup. |
| `sam_jax/training_utils/flax_training.py:515` | L2 over every parameter inside both losses, before clipping; torch optimizer weight_decay is zero. |
| `sam_jax/training_utils/flax_training.py:454,537` | Normalize each replica's gradient with no epsilon; perturb all parameters locally. Zero norm remains an invalid/non-finite case, not silently regularized. |
| `sam_jax/training_utils/flax_training.py:551,564` | Return first-pass BN, discard second-pass state, mean final gradients, then clip and update shared weights. |
| `sam_jax/training_utils/flax_training.py:615` | Average eight BN states for evaluation; count all test examples. |
| `sam_jax/datasets/augmentation.py:37` | Flip first, REFLECT pad four, random crop, then dataset normalization. |
| `sam_jax/datasets/dataset_source.py:98,192` | FP32 /255 then exact mean/std constants; train 50,000/test 10,000; drop 80 training examples each reshuffled epoch. |

[Flax 0.2.2 BatchNorm](https://github.com/google/flax/blob/v0.2.2/flax/nn/normalization.py) computes variance as E[x²]−E[x]² and updates running statistics with biased variance. Custom functional BN matches this, without PyTorch's unbiased running-variance correction. Archived wheel source for `flax/nn/base.py` and `flax/optim/momentum.py` was downloaded and checksum-verified from [PyPI](https://pypi.org/project/flax/0.2.2/) for reading only: serialization uses module-path state keys, and momentum uses v=.9v+g, update g+.9v. No legacy package was installed or executed.

[davda54 SAM](https://github.com/davda54/sam/blob/main/sam.py) was inspected only as a two-pass optimizer reference. Its epsilon, recipe, optimizer weight decay, architecture and full-batch perturbation are not copied. Functional weight dictionaries avoid mutating/restoring model parameters during perturbation.

[Compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [full example](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html): warmup and first-call cost are excluded from steady timing; fullgraph makes graph breaks fail visibly. Compiler instrumentation is recorded separately and overlapping timers are not summed.

[Channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html), [tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), and [AMP docs](https://docs.pytorch.org/docs/2.9/amp.html): memory format, GPU data, foreach/fused update and cuDNN algorithm selection are isolated candidates. BF16 autocast retains FP32 parameters and norm math; TF32/BF16 are labeled deviations. Reference disables both TF32 flags. No profiler evidence was supplied proving that historical JAX runs avoided tensor cores; that prompt statement is **unverified**, not a measured claim here.

[Official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/) supply 2.9.0 CPU/cu126 wheels. [NVIDIA CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html) specify native Linux driver 560.28.03. Setup inspects the target before selecting cu126, conservatively rejects older drivers, and refuses existing environment prefixes. Actual JS2 inspection/installation is **(to verify)**; no connection was made.

The older `run_config.md` and replication notebook retain the 200-epoch plan; the explicitly requested 50-epoch configuration follows `experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/instructions.md` and its worker. The unused 200-epoch JSON is preserved separately. RNG streams differ between frameworks; within PyTorch, method-independent epoch plans pair SAM and SGD exactly, including resume.
