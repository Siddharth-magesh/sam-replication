# A100 one-epoch SGD / SAM / IGR timing

This is notebook 08 packaged for Python/tmux. **One measured epoch each**, not the full hypothesis experiment. ResNet-18, CIFAR-100 balanced 45k train / 5k validation, batch 128, seed 42. Ten warm-up and twenty probe steps are discarded before fresh initialization and 351 measured updates. Official test images are not evaluated. Same legacy JAX/Flax updates and 100-epoch cosine horizon as notebook 08.

Methods run **sequentially** on the same idle A100. Spare VRAM does not imply spare compute; parallel methods would contaminate timings and budget projections. The launcher refuses a busy GPU and preserves the previous experiment's launcher lock. It never starts or stops a VM automatically.

## Run on csci567-sam

From local WSL, connect after the instance is active:

```bash
js2 ssh csci567-sam
```

On the instance:

```bash
cd /workspace/sam-replication
git switch repro
git pull --ff-only origin repro
python3 -B experiments/cifar100_resnet18_a100_one_epoch_timing/run.py
tmux new -s igr-timing
```

Inside tmux:

```bash
cd /workspace/sam-replication
python3 -u -B experiments/cifar100_resnet18_a100_one_epoch_timing/run.py \
  --execute --work-root /workspace/sam_igr_a100_timing \
  --reuse-env /workspace/sam_paper_replication/envs/sam \
  --planning-budget-hours 26.0812
```

Detach: Ctrl-b then d. Reattach: `tmux attach -t igr-timing`.

The existing environment is validated and **never repaired or upgraded** when `--reuse-env` is supplied. Source is cloned into the separate work root at notebook 08's pinned revision; CIFAR-100 is cached there. Nothing changes the old CIFAR-10 runtime or results. Without `--reuse-env`, a new isolated pinned environment is installed under the work root; use that only if intentionally setting up a new environment.

The shared subprocess cap is 60 minutes including setup and data preparation. A timeout may leave incomplete methods; it is not a promise of completion in one hour. CPU reporting/ZIP creation can take additional time. No automatic training resume: rerun the same command for a new attempt while reusing environment/data caches. Failed attempts are retained. Do not change batch size to hide OOM.

## Results and budget

The command prints `RESULTS` and `DOWNLOAD`. Files are under:

```text
/workspace/sam_igr_a100_timing/a100_timing_runs/<session>/
/workspace/sam_igr_a100_timing/a100_timing_runs/<session>_timing.zip
```

Read `measured_epoch_times.csv`, each method's result/status/log, `budget_projection.json`, `planning_scenario.json` and `invocation.json`. The ZIP excludes large checkpoints. A failed or missing method cannot produce a full-study estimate. Validation accuracy after one epoch is a diagnostic, not a hypothesis finding.

`26.0812` is the last known old-ledger balance, **not live allocation**; subtract later Colab/idle/other usage and set your actual planning ceiling. The default scenario projects 100 epochs x 3 seeds x 3 methods plus three additional 100-epoch IGR tuning trials, a 60-minute unmeasured evaluation reserve and 20% allowance. These are estimates only. No full-study training is launched. `invocation.json` is a separate incremental receipt; reconcile it once with project accounting. The old ledger is not overwritten. Concurrent notebook jobs do not respect this wrapper's locks, so keep other GPU work stopped.

From **local WSL**, download the printed ZIP (substitute the session filename and current instance IP):

```bash
mkdir -p /mnt/d/ML_Research/sam-replication/experimental_results/igr_timing
scp ubuntu@149.165.159.148:/workspace/sam_igr_a100_timing/a100_timing_runs/SESSION_timing.zip \
  /mnt/d/ML_Research/sam-replication/experimental_results/igr_timing/
```

After download and when no other job is running, from WSL:

```bash
js2 stop csci567-sam
js2 status csci567-sam
```

Confirm `SHELVED_OFFLOADED`. Send the ZIP for budget review before selecting the full experiment.

## Files and verification

- `run.py`: opt-in CLI, locks, idle check, orchestration and evidence export.
- `setup_runtime.py`: notebook bootstrap with read-only existing-environment reuse.
- `prepare.py`: exact notebook configuration and dataset preparation.
- `launch.py`: exact notebook method-launch checks.
- `worker.py`: byte-identical extracted notebook 08 training worker.
- `report.py`: byte-identical extracted notebook budget reporter.
- [background.md](background.md): previous T4 observations and scientific limits.

Local syntax, source identity, CLI and reporting checks passed. Actual A100 execution, fit and time remain **(to verify)**. No remote training was run during packaging.
