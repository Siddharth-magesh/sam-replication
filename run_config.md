# Run config: SAM vs SGD (first comparison)

Goal: reproduce Table 1, first row of Foret et al. (ICLR 2021): WideResNet-28-10 on CIFAR-10, test error SAM 2.7% vs SGD 3.5% (paper numbers).

## Runs

| Run | Optimizer | rho | Epochs | Seed |
|---|---|---|---|---|
| `SAM-200_seed_0` | SAM | 0.05 | 200 | 0 |
| `SGD-200_seed_0` | SGD | 0 | 200 | 0 |

Run SAM first, then SGD. More seeds only after both finish and the time fits the budget. SGD-400 is skipped for now.

## Shared settings (identical for both runs)

| Setting | Value |
|---|---|
| Model / data | WideResNet-28-10, CIFAR-10 (50,000 train, 10,000 test) |
| Batch size | 256, split into 8 sub-batches of 32 (matches the paper's 8 GPUs) |
| Learning rate | 0.1, cosine schedule over the full run, no warmup |
| Weight decay | 0.0005 |
| Optimizer details | Nesterov momentum 0.9, gradient clipping 5.0 (code defaults) |
| Augmentation | Basic (flip, pad 4, random crop) |
| Source | Paper Appendix Table 6 (200-epoch row) |

Only rho differs between the two runs.

## How to run

1. Open the replication notebook on the A100 and run preflight (`RUN_PREFLIGHT = True`). Do not continue if it fails.
2. Set `RUN_TRAINING = True` and `RUN_SELECTION = 'SAM-200_seed_0'`. Run it.
3. When it finishes, set `RUN_SELECTION = 'SGD-200_seed_0'`. Run it.
4. If a run is interrupted: `REUSE_SESSION` = the same session folder, `RESUME_EXISTING = True`.

The selected-run budget guard is already fixed: it checks only RUN_SELECTION against the remaining cumulative 40-hour worker budget, including preflight and earlier attempts. The nine-run estimate is informational only.

For script execution on csci567-sam, use `experiments/cifar10_first_comparison/` and the first-comparison instructions at the top of `experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/jetstream_runbook.md`. Setup creates the pinned Linux Python 3.8/CUDA environment on the persistent volume; do not use the system Python for the training worker. Both full A100 preflights passed in session `20261118_143021` on November 18, 2026. The 200-epoch experiments have not started; the measured planning estimate exceeds the remaining budget.

## Measured preflight and time estimate (A100)

The earlier **8-17 hour total** was an unmeasured estimate and is superseded by the November 18 preflight. It was not a 13-hour guarantee.

Session: `/workspace/sam_paper_replication/runs/20261118_143021`, worker revision `77cd455`, full A100 40 GB, both `--deterministic-cudnn` and `--deterministic-reductions`. Both full preflights passed, including numerical/BN checks, one epoch, full test-set evaluation and checkpoint restoration. No full training has started.

| Run | First measured training epoch | Evaluation | Current estimate for 200 epochs, including planning overhead and 20% allowance |
|---|---:|---:|---:|
| SAM | 677.46 s (11.29 min) | 8.40 s | 45.84 h |
| SGD | 203.47 s (3.39 min) | 7.82 s | 14.20 h |
| **Total** | | | **60.04 h** |

Recorded usage after preflight: **1.974870 h / 40 h**; remaining **38.025130 h**. The unchanged pair does not fit the current conservative estimate. Do not bypass the budget guard or silently shorten the runs.

These are extrapolations from a first epoch, not measured 200-epoch runtimes. The first epoch can contain compilation/input overhead. SGD's saved early step sample included a 22.97-second second step versus a 0.931-second p90. The full saved per-step reports must be analyzed before attributing all first-epoch overhead to every epoch; the SAM steady-state breakdown and performance impact of deterministic kernels are **(to verify)**. The instance was turned off by the user; do not start it just to repeat preflight.

GPU-memory snapshots during training: SGD **17,485 MiB**, SAM **17,493 MiB** (about **17.1 GiB**), with 100% reported GPU utilization. These are observed allocated-memory snapshots, not measured peak or minimum-memory requirements. The configured implementation has not been demonstrated on a 4 GB GPU. System RAM is not interchangeable with CUDA GPU memory.

The executed local `repro_notebooks/colab_smoke_test.ipynb` uses the same original train_step/settings/eight-lane mapping, but runs only ten batches per method and evaluates one test batch. Its elapsed times include compilation and other overhead; its final 0 MiB snapshot was taken after the workers exited. It establishes functional smoke success, not a full-run runtime or memory benchmark.

## What to report

- Final-epoch test error (epoch 200 only, never the best epoch) for both runs.
- Compare with the paper (SAM 2.7, SGD 3.5) and say how far off we are.
- Note that this is one seed (no confidence interval) and SGD was trained for 200 epochs only. The paper reports SGD as the better of its 200- and 400-epoch results, so our SGD number may be slightly higher than the paper's 3.5.
