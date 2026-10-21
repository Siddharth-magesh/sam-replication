## Code findings (`train.py`, `flax_training.py`, models, datasets)

### Flags and defaults that differ from the paper
| Finding | Consequence |
|---|---|
| `--batch_size` defaults to **128** | Paper's CIFAR runs use **256** (App. C.1). Always pass `--batch_size 256`. |
| `--weight_decay` defaults to **0.001** | Paper's WRN-28-10 / 200-epoch row uses **0.0005** (Table 6); 0.001 is the 1800-epoch row. Pass explicitly. |
| `--learning_rate` defaults to 0.1, cosine schedule, no warmup | Matches Table 6 for WRN-28-10. The schedule length is tied to `--num_epochs`, so short runs are not prefixes of long runs. |
| `--sam_rho` defaults to 0.0 and `rho == 0` runs plain SGD | Confirmed in `flax_training.py` (SAM only if `rho > 0`). The SGD baseline is `--sam_rho 0.0`. |
| `--gradient_clipping` defaults to **5.0** (global norm) | On for both SAM and SGD, applied after the cross-device mean. The paper does not mention it. |
| `--image_level_augmentations`: `none`, `basic`, `autoaugment`, `aa-only`; default `basic` | Default matches the "Basic" rows of Table 1. |
| Cutout is a separate flag: `--batch_level_augmentations cutout` | Not part of the image-level flag. |
| `--evaluate_every` defaults to 1 | Evaluates on the test set every epoch. Use a larger value for long runs. Report the **final-epoch** test error, not the best epoch. |
| Output path is auto-built: `lr_*/wd_*/rho_*/seed_*` | Different settings and seeds don't overwrite each other. |
| Training resumes from `checkpoints/` if one exists in the output path | Use a fresh `--output_dir` per experiment, or a rerun resumes from a stale checkpoint. Checkpoints save at most hourly, which allows resuming after a Colab disconnect. |

### Differences from the paper's setup
- **Test set used for evaluation.** The log says "Used test set instead of validation set". The released code holds out no validation split, although the paper tuned rho on 10% of the training set (Sec. 3.1).
- **Single device: m = 256, not 32.** `local_batch_size = batch_size // num_devices`. The paper's 8 V100s gave 32 examples per device. `sync_perturbations` defaults to False, so each device computes its own SAM perturbation (as in the paper). Paper Fig. 3 (middle) says smaller m helps SAM, so a single-GPU SAM gain may be smaller.
- **Batch norm is per device (confirmed).** `models/utils.py` builds `nn.BatchNorm.partial(use_running_average=not train, momentum, epsilon)` with no `axis_name`, so training statistics come from the local sub-batch. Paper: BN statistics and the SAM perturbation were both over 32 examples. Ours: both over 256. Splitting each batch into 8 chunks of 32 would restore both together. m-sharpness is therefore not the only difference on one GPU.
- **Seeds do not behave as the paper describes.**
  - `models/load_model.py` uses a hardcoded `PRNGKey(0)` (lines 56 and 116), so weight initialization does not depend on `--run_seed`. [verify with `sed -n 50,60p` and `sed -n 110,125p`]
  - `--run_seed` only seeds the stochastic-layer key in `flax_training.py`. Its help text says it has no influence on dataset shuffling, and WRN-28-10 has no dropout.
  - `datasets/dataset_source.py` calls `shuffle(50000)` with no seed.
  - So repeated runs share the same initial weights and differ only through unseeded data order and augmentation. The paper reports independent initialization and shuffling for each of its 5 replicas.
  - Plan: patch init and shuffle to use `run_seed`, as its own commit on `repro`.
- **SAM training metrics are measured at the perturbed weights.** Logged `train_loss` and `train_error_rate` come from the second pass at `w + epsilon`. They are not comparable to SGD's training metrics. Compare test metrics only.

### Other notes
- `jax.random.bernoulli` is replaced by a TPU-oriented `lax.rng_uniform` version that ignores the seeded key. This matters only for dropout or ShakeDrop models, but it is the first place to look if GPU training crashes.
- TensorFlow is hidden from the GPU (`set_visible_devices([], 'GPU')`) and only runs the data pipeline. Its `libcudart.so.10.1` / `libcudnn.so.7` warnings are harmless.
- Metrics are written as TensorBoard **tensors**, not scalars. Read them with `EventAccumulator.Tensors(...)` and `tensor_util.make_ndarray` (script: `analysis/read_tb.py`). Available tags: `train_loss`, `train_error_rate`, `test_loss`, `test_error_rate`, `gradient_norm`, `param_norm`, `learning_rate`.

### Questions resolved
- BatchNorm: per device, no cross-device sync (see above).
- Seeds: `--run_seed` does not control weight initialization or shuffling (see above).

### Still open
- Exact cuDNN version on disk in the env (`ls -l .../libcudnn.so*`).
- Does the doubled-epoch SGD baseline (best of 200 or 400 epochs, paper Sec. 3.1) matter for the comparison? Run SGD at both epoch counts.