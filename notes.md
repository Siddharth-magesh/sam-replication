`train.py` tells us

| Finding | Why it matters |
|---|---|
| `--batch_size` defaults to **128** | The paper's CIFAR runs use **256** (Appendix C.1). Always pass `--batch_size 256`, or you'd be comparing against a different setup. |
| `local_batch_size = batch_size // num_devices` | On your 1 GPU, each device gets all 256 examples. The paper's 8 V100s gave **32 per device**, so this confirms the m = 256 vs. m = 32 gap. |
| `--image_level_augmentations` takes `none`, `basic`, `autoaugment` or `aa-only`, default `basic` | The default already matches the "Basic" rows of Table 1. |
| Cutout is a **separate** flag: `--batch_level_augmentations cutout` | I said earlier it was probably part of the image-level flag. It isn't. |
| `learning_rate`, `weight_decay`, `sam_rho`, `run_seed` and `from_pretrained_checkpoint` are used here but defined elsewhere | They're in `flax_training.py`. The seed flag is `--run_seed`. |
| The output path is auto-built as `lr_*/wd_*/rho_*/seed_*` | Different seeds and settings won't overwrite each other. Your own run directory names only need to say dataset/model/SAM-or-SGD/epochs. |
| `tf.config...set_visible_devices([], 'GPU')` | TF never touches the GPU, as expected. |
| Hardware Bernoulli override | `jax.random.bernoulli` is replaced with `lax.rng_uniform`, which ignores the seeded key. If a model uses dropout or ShakeDrop, that randomness may not be controlled by `--run_seed`. WRN-28-10 probably doesn't use it, but check, and note it in `NOTES.md`. It's also the first place to look if training crashes on GPU, since it was written for TPU. |


- **BatchNorm:** with 8 devices and batch 256, were BN statistics computed over 32 examples per device (unsynced)? If so, the paper's "m = 32" setup also had small-batch BN. That means m-sharpness isn't the only thing that differs on one GPU.
- **Seeds:** does `run_seed` control weight init, data shuffling, and augmentation randomness?
