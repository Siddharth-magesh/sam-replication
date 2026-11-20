# Full A100 ResNet-18 / CIFAR-100 study

This is the full SGD/SAM/IGR comparison requested after the T4 intermediate and A100 timing runs. It reuses the timing experiment's installed environment, cached CIFAR-100 and pinned original ResNet block. It does not install packages, modify the authors' implementation, launch a VM or stop it automatically. Run methods sequentially on the prepared `csci567-sam` A100 with at least 38,000 MiB. The default command only prints the plan.

## Frozen protocol

| Setting | Value |
|---|---|
| Dataset | CIFAR-100; balanced 45,000 train / 5,000 validation; split seed 2026 |
| Model | CIFAR ResNet-18, 100 outputs, FP32 |
| Training | 100 epochs, batch 128; 351 updates/epoch, 72 shuffled images dropped |
| Optimizer | SGD, momentum 0.9, Nesterov off, LR 0.1 with 100-epoch cosine decay |
| Regularization | L2 weight decay 0.0005 on all parameters; final gradient global clipping at 5 |
| SAM | Absolute global L2 radius 0.05, same two-pass convention as timing worker |
| IGR tuning | Lambda 0, 0.01, 0.1; 100 epochs each on development seed 42 |
| Selection | Highest final validation accuracy; ties smallest lambda; no test use |
| Final runs | Fresh training with selected lambda; SGD/SAM/IGR each on seeds 0, 1, 2 |
| Final weights | Epoch 100, never best-test or best-validation checkpoint |
| Clean evaluation | All 10,000 official test images |
| Shared probe | Fixed balanced 1,000 test images, 10/class, seed 31415 |
| Input sensitivity | Mean per-image CE gradient L2 and squared L2, raw [0,1] pixels |
| PGD | Untargeted CE, L-infinity 8/255, step 2/255, 20 steps, 3 seeded restarts |
| Sharpness | CE increase under absolute global L2 radii 0.025/0.05, 10 ascent updates, 3 random sphere starts |

Initialization, epoch shuffles and augmentation are paired within each final seed. The first-epoch order and initial model/BN hashes are checked across methods. Random attack/weight starts are shared across methods within each seed. BN is frozen for IGR's penalty and all inference measurements; clean training updates BN. SAM retains the first pass BN update. IGR differentiates through per-image input gradients into parameters. Gradient clipping, weight decay and BN conventions match the tested timing worker; this is not the paper's original WRN experiment.

All three development trials and all nine final training runs finish before test evaluation starts. Selection is written once and checked on resume. If lambda 0 wins, retain that outcome; do not force a nonzero choice. No SAM+IGR arm or architecture sweep is included. Final evaluation has stronger probes than the T4 pilot, so its robustness/sharpness values are not directly interchangeable with the earlier 100-image PGD-10/three-step values.

## Launch on the course machine

Reviewed against the successful `20261119_153717_b8692b` timing setup on November 19. No performance flags or scientific functions were changed in this review. This implementation is already the fast measured ResNet configuration, not the slower WRN reproduction. Keep sequential execution: spare VRAM does not provide independent GPU compute. The added operational checks do not certify full training or stronger evaluation; those remain **(to verify)**.

From **local Ubuntu/WSL** (not the remote Ubuntu terminal), if the instance is currently shelved:

```bash
js2 start csci567-sam
js2 ssh csci567-sam
```

Only start this instance when ready to run. A100 allocation rate is 64 SUs/hour in the course guide: the 15.02-hour estimate is approximately 962 SUs; a 20-hour run could use 1,280 SUs, plus idle/startup time. Check the actual remaining allocation before unshelving. These are planning figures, not a cloud billing query.

First confirm the machine is idle and the team's actual remaining GPU allocation covers this study. The prior declared 26.0812 hours was not a live balance. Account for the timing invocation and other usage once. The study journal records incremental invocation time; it does not overwrite the old team ledger and excludes VM idle time.

```bash
cd /workspace/sam-replication
git switch repro
git pull --ff-only origin repro
python3 -B experiments/cifar100_resnet18_sam_igr_full/run.py
tmux new -s igr-full
```

Inside tmux:

```bash
cd /workspace/sam-replication
python3 -u -B experiments/cifar100_resnet18_sam_igr_full/run.py \
  --execute \
  --work-root /workspace/sam_igr_a100_timing \
  --reuse-env /workspace/sam_paper_replication/envs/sam \
  --budget-hours 20
```

Use the same `--reuse-env` path that passed the timing run. If timing created its own environment, use `/workspace/sam_igr_a100_timing/envs/sam`. The work root must already contain the pinned `work/sam`, `cuda` libdevice and prepared TFDS cache from timing. Validation fails rather than repairing an environment or downloading a dataset. Requirements remain unchanged.

Detach with Ctrl-b then d; reattach with `tmux attach -t igr-full`. The launcher prints `STUDY` immediately and appends epoch progress to each job's `train.log`. Default cumulative incremental cap is 20 hours, including failed invocations; it is frozen in the manifest. It is a ceiling, not a promise to use 20 hours. The supplied epoch timings imply approximately 8.13 hours for final training, 3.39 for tuning, plus an unmeasured one-hour evaluation reserve and 20% allowance: approximately 15.02 hours. Stronger evaluation has not been timed; it can exceed that reserve. No new A100 training has been executed during implementation.

The first jobs are IGR tuning trials, not SGD. It is normal to see `tune_lambda_0.0`, `tune_lambda_0.01`, then `tune_lambda_0.1`. After selection, nine final training runs execute sequentially, then nine evaluations. Expect epoch lines after the initial compile; the launcher prints heartbeat messages while subprocesses work.

The current study path is also saved to `/workspace/sam_igr_a100_timing/latest_full_study.txt`. A normal relaunch refuses to create a duplicate while that recorded study is unfinished.

In a **second SSH terminal**, monitor without launching another job:

```bash
STUDY=$(cat /workspace/sam_igr_a100_timing/latest_full_study.txt)
cat "$STUDY/progress.json"
find "$STUDY" -name history.json -exec tail -n 18 {} \;
```

`progress.json` appears when the first job starts. To follow the first job live:

```bash
tail -n 30 -F "$STUDY/tune_lambda_0.0/train.log"
```

Use the job/phase listed in `progress.json` for later logs (`train.log` or `evaluate.log`). Avoid pulling/changing scripts after creating the study; source fingerprints intentionally freeze resume compatibility.

The launcher shares locks with the timing and earlier WRN launcher and refuses an already busy GPU. Notebook jobs do not honor these locks, so do not start other GPU work concurrently. GNU `timeout` enforces the remaining invocation allowance even if the parent process dies. Interrupt with Ctrl-C for a graceful stop. Do not delete partial evidence.

## Numerical failure on November 19: diagnostic first

Study `20261119_200309_788f36` stopped in lambda 0.01 epoch 75 with a non-finite training metric; epoch 74 was saved. Lambda 0 finished at 76.94% validation accuracy. No final experiments ran. Do not resume this numerical failure unchanged or remove the finite checks.

The diagnostic replays epoch 75 from the same saved optimizer/BN checkpoint twice: legacy Flax BN and an opt-in centered-variance BN candidate. The latter computes mean squared centered activations instead of subtracting two large second moments. It preserves BN names, initialization, momentum and epsilon; negative saved running variances are reported, never silently clipped. This is a candidate numerical correction, **not a confirmed root cause or verified full-run repair (to verify)**. The IGR train/inference BN mismatch also remains a possible source of poor validation accuracy.

After these files are available on the machine, run only this diagnostic inside your existing tmux session:

```bash
cd /workspace/sam-replication
git pull --ff-only origin repro
python3 -u -B experiments/cifar100_resnet18_sam_igr_full/diagnose.py \
  --study-dir /workspace/sam_igr_a100_timing/full_runs/20261119_200309_788f36 \
  --execute
```

It has a 15-minute total ceiling, uses the existing environment/data, and charges the existing study ledger. Without `--execute` it only prints a plan. Results go into `STUDY/diagnostics/TIMESTAMP/`: `comparison.json`, per-variant logs, `diagnostic_before.json`, and either `diagnostic_result.json` or `numerical_failure.json`. The original checkpoint, histories, frozen source snapshots and results remain untouched. No final training launches. Preserve the entire diagnostic folder; a finite replay with chance-level validation is not a successful repair. GPU/checkpoint compatibility and stability remain **(to verify)**. After diagnosis, any adopted BN change needs a documented new protocol and retained prior budget, not mixed old/new tuning results.

## Resume and budget recovery

Checkpoints include optimizer/momentum, model weights, BN state and completed epoch. Each completed epoch is committed atomically; a partial epoch is replayed using the same deterministic augmentation generator. GPU nondeterminism means bitwise equality of a resumed trajectory is not guaranteed. Source hashes, environment prefix, runtime revision, GPU identity, protocol and original cap must match. Keep the same checkout version after starting a study.

```bash
python3 -u -B experiments/cifar100_resnet18_sam_igr_full/run.py \
  --execute --work-root /workspace/sam_igr_a100_timing \
  --reuse-env /workspace/sam_paper_replication/envs/sam \
  --budget-hours 20 \
  --study-dir /workspace/sam_igr_a100_timing/full_runs/YOUR_STUDY_ID
```

Completed trials/runs are reused and incomplete epochs replayed. Evaluation resumes by clean/sensitivity/PGD/sharpness unit; an interrupted unit restarts, without retraining. If the cumulative cap is exhausted, preserve the incomplete study and report its status; changing the frozen cap requires a separately documented protocol/budget decision rather than silently resetting the journal.

For a study created by the reviewed launcher, the simpler resume command below uses its saved source snapshot, protecting against later repository updates. Run inside tmux after the previous process has exited:

```bash
STUDY=$(cat /workspace/sam_igr_a100_timing/latest_full_study.txt)
python3 -u -B "$STUDY/run.py" --execute --budget-hours 20 --resume-latest
```

The 20-hour cap is cumulative, not reset on resume. For older studies without the pointer, use the explicit `--study-dir` command above and the original saved launcher version. Do not repeat a fresh launch after a timeout.

An uncatchable host crash leaves a reservation of the remaining allowance. Wait for all bounded child processes to stop and independently verify the invocation's actual active hours from external timestamps/logs. Settle that receipt exactly once, with **verified actual hours**, before the normal resume command:

```bash
python3 -B experiments/cifar100_resnet18_sam_igr_full/run.py \
  --execute --budget-hours 20 \
  --study-dir /workspace/sam_igr_a100_timing/full_runs/YOUR_STUDY_ID \
  --settle-interrupted-hours VERIFIED_HOURS
```

Include the same custom work-root/reuse-env arguments if applicable. This command only reconciles the interrupted receipt and does not train. Preserve the evidence supporting the manual value. Do not settle a live invocation, overwrite the journal or count the receipt twice in team accounting.

## Outputs and interpretation

Outputs are outside this repository under `/workspace/sam_igr_a100_timing/full_runs/<study>/`. `DOWNLOAD` points to `<study>_evidence.zip`; it includes source snapshots, manifest/configs, split/probe indices, selection, all histories/statuses/logs, evaluation trajectories, time receipts and reports. Large optimizer checkpoints stay on persistent storage and are excluded from ZIP. Download the evidence before stopping the VM; copy checkpoints separately if you need off-machine recovery.

- `per_seed_metrics.csv`: actual completed evaluation results only.
- `mean_std_metrics.csv`: descriptive mean/sample standard deviation, only for methods with all three seeds.
- `paired_differences.csv`: paired seed differences and marginal t95 intervals, only when all nine evaluations finish. These intervals are not corrected for multiple comparisons and are imprecise with three seeds.
- `summary.json`: completeness and methodological limits; it never declares either hypothesis confirmed.
- `lambda_selection.json`: every tuning score and frozen choice, including lambda 0.
- `incremental_budget.json`: active invocation receipts and conservative crash reservations.

Reports can also be regenerated without a GPU or the legacy environment:

```bash
python3 -B experiments/cifar100_resnet18_sam_igr_full/report.py \
  /workspace/sam_igr_a100_timing/full_runs/YOUR_STUDY_ID
```

The weight proxy is a lower bound from finite optimization, with unregularized inference CE, fixed BN and all trainable parameters including BN affine terms. It includes the unperturbed candidate so the reported increase is nonnegative. Its absolute radius is sensitive to parameterization; trajectories and weight norms are retained. PGD counts an example robust only if it is clean-correct and correct at every visited iterate in every restart; this is not a robustness certificate or a complete masking check.

Download from **local WSL**, using the current instance IP and replacing `STUDY_ID` with the printed ID:

```bash
mkdir -p /mnt/d/ML_Research/sam-replication/experimental_results/igr_full
scp ubuntu@149.165.159.148:/workspace/sam_igr_a100_timing/full_runs/STUDY_ID_evidence.zip \
  /mnt/d/ML_Research/sam-replication/experimental_results/igr_full/
```

After verifying the download and confirming no GPU job is active, from local WSL:

```bash
js2 stop csci567-sam
js2 status csci567-sam
```

Wait for `SHELVED_OFFLOADED`. Do not delete the persistent study/checkpoints. Failure also leaves the VM active; collect evidence and shelve it if not promptly resuming.

This experiment tests whether IGR provides an alternative route to generalization in this setup. Similar IGR/SAM accuracies do not establish equivalence without a predeclared margin and adequate precision, and do not establish that input stability causes SAM's own benefit. If SAM fails to improve on SGD here, there is no demonstrated SAM gain in this setup for IGR to reproduce. The T4 intermediate's negative SAM result must remain reported alongside any later positive results. The team should write its competing predictions before looking at these new test results and report inconclusive/negative outcomes honestly.

Local syntax, numerical-update source identity, control-flow recovery and report checks are documented in project notes. Full GPU execution and interrupted GPU-training recovery remain to verify. The launcher leaves the instance running on success/failure. Download evidence, reconcile charges once, then manually stop/shelve the machine when no other job is using it.
