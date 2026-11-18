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

For script execution on csci567-sam, use `experiments/cifar10_first_comparison/` and the first-comparison instructions at the top of `docs/jetstream_runbook.md`. Setup creates the pinned Linux Python 3.8/CUDA environment on the persistent volume; do not use the system Python for the training worker. GPU execution remains (to verify).

## Time estimate (A100)

These are estimates, not measurements. Replace them with the preflight timings.

| Run | Estimated time |
|---|---|
| SAM, 200 epochs | ~5 to 11 h |
| SGD, 200 epochs | ~3 to 6 h |
| **Total** | **~8 to 17 h** |

## What to report

- Final-epoch test error (epoch 200 only, never the best epoch) for both runs.
- Compare with the paper (SAM 2.7, SGD 3.5) and say how far off we are.
- Note that this is one seed (no confidence interval) and SGD was trained for 200 epochs only. The paper reports SGD as the better of its 200- and 400-epoch results, so our SGD number may be slightly higher than the paper's 3.5.
