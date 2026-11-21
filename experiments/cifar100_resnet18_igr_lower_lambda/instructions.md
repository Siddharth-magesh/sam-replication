# Exploratory lower-lambda IGR extension

This follows completed v3 study `20261120_043937_4863bf` and its post-hoc validation sensitivity diagnostic. Previous lambda 0.01/0.1 reduced local input CE-gradient sensitivity but lost about 19 validation accuracy points. This extension checks three gentler penalties. It preserves all previous findings and is explicitly exploratory because prior test results have already been seen.

## Protocol fixed before execution

| Setting | Rule |
|---|---|
| New lambda grid | 0.00001, 0.0001, 0.001; no adaptive expansion |
| Development | Each candidate trains from scratch for 100 epochs, seed 42 |
| Selection | Highest final-epoch accuracy on all 5,000 validation images **among these positive candidates**; exact ties choose smaller lambda |
| Lambda zero | Existing seed-42 epoch-100 validation score is a reference, not a candidate in this active-intervention extension |
| Final runs | Selected positive lambda retrained from scratch for 100 epochs on seeds 0, 1, 2, even if it loses to lambda zero in development |
| Reference arms | Reuse completed v3 SGD/SAM final seeds 0, 1, 2, with provenance; do not retrain them |
| Dataset/model | Same CIFAR-100 split, CIFAR ResNet-18, FP32, batch 128 |
| Optimizer | Same SGD momentum 0.9, LR 0.1 cosine over 100 epochs, weight decay 0.0005, clipping 5 |
| BatchNorm | Same v3 centered-moment BN and saved evaluation-only 100-training-batch recalibration |
| Validation diagnostic | Same balanced 1,000 validation images and probe seed 271828 as the previous diagnostic; accuracy, mean input-gradient L2 and squared L2 |
| Final evaluation | Same full 10,000-image test accuracy; shared 1,000-image test probe for sensitivity, PGD-20/3 restarts and sharpness radii 0.025/0.05, 10 ascent updates/3 starts |

The positive-only rule is a deliberate change from the original search: we need an active IGR intervention. It does **not** mean the selected method is better than lambda zero. `lambda_selection.json` always includes the zero reference, the validation accuracy difference and whether the selected candidate beat that reference. Selecting an active configuration that performs poorly is a valid outcome and is retained. Sensitivity measurements do not select lambda. If all candidates fail numerically, no final configuration is selected. Failed trials remain recorded. Other errors stop execution.

All three final training runs finish before any new official-test evaluation. Final epoch weights are used, never best-test weights. Initial model/BN and first-epoch data-order hashes must match the old SGD/SAM references within each seed. Original v3 worker/runtime snapshots are verified by hash and reused byte-for-byte. The original full runner, snapshots, journals, checkpoints, histories, selection and metrics are not modified.

This extension is not an independent confirmatory experiment: its design follows previous test observations, and its references are reused. No formal accuracy equivalence margin or automatic hypothesis verdict is introduced. The finite-gradient toy check and original checkpoint recovery remain available through the unchanged worker. Lower CE-gradient sensitivity is not a prediction-consistency or robustness guarantee.

## Prepare and run inside the A100 machine

Publish these changes, then on the prepared course machine:

```bash
cd /workspace/sam-replication
git pull --ff-only origin repro
python3 -B experiments/cifar100_resnet18_igr_lower_lambda/run.py
```

This prints the fixed plan without GPU work or writes. Verify the machine is idle and the actual team allocation covers the new work. A planning estimate is approximately **5.5 hours**: six roughly 40-minute active-IGR training runs, 0.5 unmeasured evaluation hours, and 20% allowance. These lower lambdas have not been timed. The default cap is **8 incremental hours for this extension**, not a reset of prior usage or a live team balance. Previously recorded study time remains separate and must be included once in total project accounting; idle allocation is additional.

Inside tmux, execute:

```bash
tmux new -s igr-lower-lambda
```

```bash
cd /workspace/sam-replication
python3 -u -B experiments/cifar100_resnet18_igr_lower_lambda/run.py \
  --reference-study /workspace/sam_igr_a100_timing/full_runs/20261120_043937_4863bf \
  --budget-hours 8 \
  --execute
```

The reference directory must be the full original completed study, including source/config/train-result records and final reference metrics; the small curated GitHub result folder is insufficient. Its pinned runtime, original source checkout, CIFAR-100 cache and CUDA libdevice must still be available at their recorded paths. New packages are not installed and the VM is neither started nor stopped automatically. Shared locks block concurrent timing/WRN/full/probe jobs, and a busy GPU is rejected. Notebook jobs do not honor these locks; keep other GPU work stopped.

Detach with Ctrl-b then d. The launcher prints `EXTENSION STUDY` immediately. Results live outside Git under:

```text
/workspace/sam_igr_a100_timing/lower_lambda_runs/<extension-id>/
```

`latest_lower_lambda_study.txt` records that path; `latest_full_study.txt` is unchanged. Each new job has `train.log`, epoch history and atomic optimizer/momentum/BN checkpoints. `progress.json` names the current training/evaluation job. Validation diagnostic logs are under each tuning job's `validation_probe/`. Initial compilation may delay the first epoch print; subprocess heartbeats continue.

## Resume without resetting the experiment

Stop gracefully with Ctrl-C if necessary. A partial epoch is replayed; completed epochs and jobs are reused. The original source fingerprints, reference evidence and cap must match. GPU nondeterminism prevents promising bitwise-identical replay. Use the saved extension runner rather than a newer checkout:

```bash
EXTENSION_STUDY=$(cat /workspace/sam_igr_a100_timing/latest_lower_lambda_study.txt)
python3 -u -B "$EXTENSION_STUDY/run.py" \
  --reference-study /workspace/sam_igr_a100_timing/full_runs/20261120_043937_4863bf \
  --study-dir "$EXTENSION_STUDY" \
  --budget-hours 8 --execute
```

The extension journal accumulates invocation time and reserves its remaining allowance before execution. GNU timeout bounds subprocesses even after parent loss. An uncatchable host crash leaves a conservative reservation; after all children stop, verify actual active time from external evidence and settle once:

```bash
python3 -B "$EXTENSION_STUDY/run.py" \
  --reference-study /workspace/sam_igr_a100_timing/full_runs/20261120_043937_4863bf \
  --study-dir "$EXTENSION_STUDY" --budget-hours 8 \
  --settle-interrupted-hours VERIFIED_ACTUAL_HOURS --execute
```

Settlement only reconciles the receipt; it does not train. Keep supporting timing evidence. If the cap is exhausted, preserve the incomplete study rather than silently raising or resetting the frozen cap. Validation and final measurements resume by metric unit; interrupted units may be recomputed. Do not change original evidence or source scripts during a study. Running again without `--study-dir` is blocked while the latest extension is unfinished.

## Review the results

The launcher prints `RESULTS` and `DOWNLOAD`. `<extension-id>_evidence.zip` includes source snapshots, manifests, all tuning scores/failures, per-epoch histories, validation diagnostics, original reference provenance, logs and reports. Large checkpoints remain in persistent storage and are excluded from the ZIP.

- `lambda_selection.json`: fixed positive-only rule, every completed/rejected candidate and comparison with the prior lambda-zero validation reference.
- `tuning_diagnostics.csv`: full-validation scores and separate 1,000-image sensitivity diagnostics. Full-set and subset accuracies are different metrics.
- `per_seed_metrics.csv`: actual new IGR metrics and explicitly labeled reused SGD/SAM metrics.
- `mean_std_metrics.csv`: descriptive means/sample SDs for methods with all three seed results.
- `paired_differences.csv`: IGR minus each reference, with marginal t95 intervals when all final evaluations finish. Three seeds yield imprecise intervals; no multiplicity correction or equivalence claim.
- `reference_provenance.json` and per-reference `reference_only.json`: original source/hashes and `retrained=false` labels.
- `incremental_budget.json`: only this extension's active invocation receipts. Add once to prior project usage; it does not cover VM idle time.
- `summary.json`: completeness and explicit scientific limitations, never an automatic causal conclusion.

No more candidates, epochs, seeds or success criteria are added automatically. Report whichever outcome occurs, alongside the original stronger-lambda negative results. If active IGR improves accuracy with lower sensitivity and higher measured sharpness than SAM, that would support an alternative route to generalization in this setup; it would not identify SAM's own causal mechanism. If not, retain the negative/inconclusive result without expanding the grid until it produces a preferred story.

Local Python 3.8 syntax/dry-run, source snapshot identity, selection/tie/failure cases, read-only reference/configuration guards, standalone saved-runner dry-run, mixed-origin reports and known paired-difference calculations were checked. Numerical training/evaluation of this new grid remains **to verify**. No GPU job was launched during implementation. Download evidence and manually shelve the instance when no other work is using it.
