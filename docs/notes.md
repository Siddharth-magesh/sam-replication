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

## Completed ResNet-18 benchmark and pilot (2026-11-13)

The original task is complete. Executed notebooks are `repro_notebooks/02_cifar10_resnet18_benchmark_pilot.ipynb` (ordinary momentum) and `repro_notebooks/03_cifar10_resnet18_nesterov_followup.ipynb` (Nesterov follow-up). Notebook bytes, including all outputs, are preserved from the supplied executed files. Per user instruction this intentionally overrides the usual cleared-output policy. Final artifacts are in `results/momentum_sgd/` and `results/nesterov_followup/`; provenance records notebook/worker hashes. Embedded workers exactly match the executed workers, and phase markers match worker/config hashes. All methods have epochs 1 through 10. No new GPU experiments were run during packaging.

| Run | SGD ms/step | SAM ms/step | FGSM ms/step | SGD final val % | SAM final val % | FGSM final val % | Pilot wall minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary momentum | 109.00 | 247.76 | 203.94 | 89.24 | 87.82 | 79.12 | 35.20 |
| Nesterov | 96.54 | 190.52 | 166.14 | 89.70 | 88.28 | 80.06 | 31.70 |

Each run uses one Tesla T4, batch 128, 20 warm-up and 100 timed steps per method. Timing synchronizes outputs and includes augmentation/transfers, excluding compilation/setup. Compare speed within each run: separate Colab sessions have different timing conditions, so their latency difference cannot be attributed to Nesterov. See benchmark_summary.csv for projections and final_epoch_summary.csv for actual training/evaluation cost. Pilot wall time includes overhead.

Setup: pinned Python 3.8 / JAX 0.2.6 / jaxlib 0.1.57 CUDA 11 / Flax 0.2.2. CIFAR ResNet-18, 3x3 stride-1 stem without max-pool, 11,173,962 parameters. Seed 42, stratified 45k/5k training/validation split, official test set unused. Shared initialization, shuffle and augmentation across methods, 351 full batches per epoch. LR 0.1, momentum 0.9, weight decay 0.0005, clipping 5, FP32, identical 10-epoch cosine schedule. Near-zero final training LR is valid; validation itself does not use an LR.

SAM rho 0.05 uses global L2 perturbation and released-code regularization in inner/outer gradients, with first-pass BatchNorm state retained. FGSM epsilon 4/255 attacks raw pixels before normalization, clips to [0,1], stops attack gradients and trains on 100% adversarial inputs, retaining clean-pass BatchNorm state. Clean validation accuracy does not measure robustness.

Nesterov diagnostics passed without warnings: zero-perturbation checks, finite/range/budget checks, SAM traced operations and constants, and numerical reference tolerance. The optional section 4B probe was manually interrupted (KeyboardInterrupt retained in notebook); it did not block the completed benchmark/pilot. Skip section 4B and its download cell on normal reruns. Both notebooks retain RUN_PILOT=True from their executed runs: inspect the estimate and set False if only benchmarking. Setup may restart the Colab runtime; then continue setup and execute configuration before the worker cell. Rerun configuration to create a fresh output folder. Upload either notebook directly to Colab; its setup clones the repository automatically. No user-side clone is necessary.

Paper comparison: supplied 2010.01412v3.pdf, Section 3.1/Table 1/Appendix C.1. WRN-28-10 Basic uses 200 epochs, CIFAR batch 256 across eight GPUs (m=32), test error and five replicas; Table 1 reports SAM 2.7 +/- 0.1% vs SGD 3.5 +/- 0.1%. Our single-seed, single-GPU, 10-epoch ResNet-18 validation pilots are not that replication. SAM finishes 1.42 percentage points below SGD in both supplied runs; no code or settings were changed to reverse that result. Released SGD uses Nesterov whereas the original task used ordinary momentum. This follow-up does not establish the cause of the gap.

Historical attempts: the earlier ordinary-momentum pilot reached SGD 89.36%, SAM 88.02%, FGSM 79.52%. Earlier audit attempts failed zero-rho/reference comparisons, a partially created environment skipped dependency installation, repeated append/patch cells duplicated worker code, and a runtime restart exposed a missing ast import. A probe also observed same-function execution variability. Consolidation replaced append/patch cells with one worker string, setup now installs dependencies even when Python already exists, and clean-v4 records numerical-reference warnings conditional on matching traces/constants while retaining other checks. The final supplied follow-up passed numerical tolerance too. Failed attempts are recorded here; obsolete notebooks, downloaded clones, redundant run folders and verbose debug/probe dumps were removed at the user's request. Current results retain diagnostic differences, configuration, environment, phase markers, timings, epoch histories and figures; workers remain embedded in notebooks. Raw logs, dataset split archives and checkpoint files are not part of the final small-results package.
