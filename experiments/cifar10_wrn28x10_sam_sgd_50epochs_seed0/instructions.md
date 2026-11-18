# WRN-28-10 / CIFAR-10: SAM-50 and SGD-50, seed 0

50 epochs **each**, batch 256, eight logical sub-batches, cosine schedule over 50 epochs. Reduced experiment, not the paper's 200-epoch result. Reuses the installed Python 3.8 environment and cached dataset. Both strict preflights run once for this new configuration. Approximately 15 hours for the pair is an estimate; the invocation cap is 18 hours. No automatic shelving.

Connect from WSL to **csci567-sam only**:

```bash
js2 ssh csci567-sam
```

On the instance (existing checkout):

```bash
cd /workspace/sam-replication
git switch repro
git pull --ff-only origin repro
python3 experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/run.py
tmux new -s sam50
```

Inside tmux, launch Python:

```bash
cd /workspace/sam-replication
python3 -u experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/run.py --execute --max-hours 18
```

Detach with **Ctrl-b, then d**; this lets Python continue if the laptop disconnects. Reconnect/monitor:

```bash
tmux attach -t sam50
```

If interrupted, run inside tmux (same checkpoints, no completed work repeated):

```bash
cd /workspace/sam-replication
git pull --ff-only origin repro
python3 -u experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/run.py --execute --resume --max-hours 18
```

If a strict preflight fails, inspect its log before retrying; partial preflights require a fresh session (omit `--resume`). Do not delete the budget ledger. Failed workers return to the shell; they do not shut down the VM.

Results/checkpoints: the folder printed as `SESSION`, also recorded here:

```bash
cat /workspace/sam_paper_replication/latest_50epoch_session.txt
```

`run.py` orchestrates; `worker.py` wraps the authors' training; `supervisor.py` enforces the shared 40-hour budget; `launch.py` handles logs/recovery; `report.py` saves final-epoch CSV and curves. The three supplied Colab setup failures are charged once, without adding their prior baseline again.

After completion and saving results, from WSL:

```bash
js2 stop csci567-sam
```
