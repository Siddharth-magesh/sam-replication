# SAM replication on Jetstream2: runbook and chat handoff

Updated: November 17, 2026. Read this entire document before running GPU commands.

## 1. What we are reproducing

The target is Table 1's first WRN-28-10/CIFAR-10 row in Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization* (ICLR 2021), paper 2010.01412v3. The empirical target is lower final test error with SAM; this experiment does not establish that flatness causes generalization.

| Item | Fixed experiment |
|---|---|
| Model | Authors' WideResNet-28-10 |
| Dataset | CIFAR-10: all 50,000 training and 10,000 official test examples |
| Methods and duration | SAM-200, independent SGD-200, independent SGD-400 |
| Seeds | 0, 1, 2; nine total runs |
| Batch | 256, split into eight logical sub-batches of 32 |
| Optimizer | Original Nesterov momentum 0.9; learning rate 0.1; duration-specific cosine schedule |
| Other settings | Weight decay 0.0005; gradient clipping 5; SAM rho 0.05; FP32 |
| Augmentation | Original basic augmentation; no cutout or AutoAugment |
| Software | Python 3.8, JAX 0.2.6, jaxlib 0.1.57+cuda110, Flax 0.2.2, CUDA 11.0, cuDNN 8.0.5 |
| Paper reference | SAM 2.7%, SGD 3.5% test error; paper uses five replicas |
| Our differences | Three seeds; eight logical replicas on one A100 instead of eight physical V100 GPUs |

The runner imports the original SAM code at pinned fork commit `f7b9ff46ee0deab8004d5508baf68b7e7c3dad1a`. The launcher branch may be newer; the runtime source checkout is deliberately pinned. Do not update the old libraries to make an error disappear. The existing absl-py compatibility adjustment is applied only to a runtime requirements copy.

## 2. Current readiness: do not skip this

- The combined T4 smoke did **not** pass. SGD passed ten updates, finite losses, parameter changes, BatchNorm checks, evaluation and checkpoint restoration at steps 5 and 10.
- SAM ran out of GPU memory at the eight-logical-replica diagnostic on T4, including after discarded comparison buffers were freed.
- The original pmap step itself varied between repeated calls on the same input. The smoke records numerical equivalence as inconclusive when its reference is not repeatable; functional success is not numerical proof. The main A100 preflight retains the strict numerical gate.
- A legacy checkpoint error was fixed by serializing the completed-epoch marker as a Python int instead of numpy.int32.
- The latest inspected csci567-sam flavor was g3.medium, a **10 GB A100 slice**. The background runner requires a full **40 GB A100**. Changing the machine's flavor and driver compatibility must be handled separately; no script resizes or creates an instance.
- Previous SSH checks timed out. The user subsequently reported connecting successfully. Confirm current access rather than repeating old assumptions.
- Python/shell syntax, notebook selectors, Jupyter JSON/schema and plan-only checks passed. The new background runner, full A100 preflight, runtime feasibility and automatic shelving are **(to verify)**. No full replication has been completed.

## 3. Which file does what

| File | Purpose | How it is used |
|---|---|---|
| `experiments/start_jetstream.sh` | Creates detached tmux session `sam-wrn` and captures console output | Normal background entry point; invoke with bash |
| `experiments/jetstream_wrn.py` | Configures paths, imports budget, checks hardware/identity, runs requested phase and requests shelving | Called by the shell launcher; without `--execute`, prints a plan only |
| `repro_notebooks/04_wrn28_10_cifar10_paper_replication.ipynb` | Experiment contract, original-step wrapper, independent budget supervisor, checkpoint/resume, estimates and figures | Required by the Python runner; also usable interactively; switches remain disabled in the saved notebook |
| `repro_notebooks/01_sam_colab_setup.ipynb` | Historical original setup and recorded smoke output | Preserved reference; not required by the runner |
| `repro_notebooks/02_cifar10_resnet18_benchmark_pilot.ipynb` | Completed ordinary-momentum pilot | Preserved with executed outputs |
| `repro_notebooks/03_cifar10_resnet18_nesterov_followup.ipynb` | Completed Nesterov follow-up | Preserved with executed outputs |
| `sam_jax/` | Authors' implementation | Unchanged; runtime uses a separate pinned checkout under WORK_ROOT/work/sam |
| `requirements.txt` | Original dependency pins | Unchanged |
| `docs/jetstream_runbook.md` | This runbook | Give this file to another chat or teammate |
| `docs/notes.md` | Technical findings and failed/successful attempt history | Preserve failures and deviations |
| `docs/project_tracking.md` | Task status and AI-usage history | Update after verified runs |

Clone the **whole repository**. Copying only the two launch scripts is insufficient because they read the WRN replication notebook cells at runtime. The previous temporary WSL file `/tmp/sam_js2_stage/setup_and_preflight.py` is superseded by the tracked launchers and is not needed.

The repository is consolidated on `repro`, its default branch. The obsolete `00_colab_smoke_test.ipynb` and `initial_raw_testing.ipynb` were removed from the current tree; their historical versions remain recoverable in Git. Setup was moved into `04_wrn28_10_cifar10_paper_replication.ipynb` before removal. Completed notebooks 01/02/03 retain their original bytes and outputs.

## 4. Machine, storage and budgets

Only `csci567-sam` is authorized. Its current OpenStack UUID is `195e6a8f-8ae1-45be-bcd7-7845b0b2419f`. The runner verifies the instance metadata UUID and checks that the cloud credentials resolve that UUID to this name. Never substitute another team's instance. If this instance is rebuilt with a new UUID, stop and review identity before changing the guard.

| Location on the machine | Contents |
|---|---|
| `/workspace/sam-replication` | Git clone containing the launcher and notebooks |
| `/workspace/sam_inputs` | Imported ledger and optional verified CIFAR archive |
| `/workspace/js2-control` | Separate modern Python environment for OpenStack lifecycle commands |
| `/workspace/sam_paper_replication/envs/sam` | Pinned Python 3.8 experiment environment |
| `/workspace/sam_paper_replication/work/sam` | Pinned authors' source checkout |
| `/workspace/sam_paper_replication/tensorflow_datasets` | Dataset cache |
| `/workspace/sam_paper_replication/gpu_budget.json` | One authoritative cumulative worker-time ledger |
| `/workspace/sam_paper_replication/runs/SESSION` | Worker, config, manifests, logs, checkpoints and summaries |
| `/workspace/sam_paper_replication/launcher_logs` | Background console logs |
| `/workspace/sam_paper_replication/launcher_*.json` | Launcher outcome and shelving-request receipts |

There are **two different budgets**. The course allocation has a 10,000-SU cap, with usage shared by the team. The guide quotes 64 SUs/hour for a full A100; an allocated idle machine also consumes credits. Separately, this experiment has a cumulative 40-worker-GPU-hour cap, including failed attempts and preflight. Its last transferred ledger records 0.8682869146 hours used and 39.1317130854 remaining; re-read the live ledger after every run. Setup and idle allocated time are additional allocation costs.

Shelving stops compute billing; Linux shutdown/SHUTOFF is not the same. Automatic shelving requires authenticated cloud control on the VM, since a sleeping laptop cannot issue the stop command. No secrets, datasets, raw logs, ledger or checkpoints belong in Git. No script touches another instance or deletes the target.

## 5. Prepare the server and inputs


Use the consolidated `repro` branch. Log into **csci567-sam** and clone on the volume:

```bash
cd /workspace
git clone --branch repro https://github.com/Siddharth-magesh/sam-replication.git
cd /workspace/sam-replication
nvidia-smi
```

If this checkout already exists, inspect its branch/status first; do not overwrite it. Use a persistent mounted `/workspace`. Do not run `js2 setup`: its general PyTorch/vLLM environment is unnecessary for this legacy JAX experiment.

For automatic shelving after laptop disconnection, the instance needs an independently configured OpenStack CLI and your cloud credentials **outside Git**. On the instance:

```bash
python3 -m venv /workspace/js2-control
/workspace/js2-control/bin/python -m pip install python-openstackclient
mkdir -p ~/.config/openstack
chmod 700 ~/.config/openstack
# Securely transfer your existing clouds.yaml to ~/.config/openstack/clouds.yaml.
chmod 600 ~/.config/openstack/clouds.yaml
export OS_CLOUD=CIS262089_IU
/workspace/js2-control/bin/openstack server show 195e6a8f-8ae1-45be-bcd7-7845b0b2419f -c name -c status
```

This control environment is separate from the pinned SAM environment. Do not commit credentials. Confirm the response names csci567-sam. If the cloud name differs in your credential file, use that name. The runner checks credentials before starting work. Setup/access failures before that check cannot self-shelve: use `~/bin/js2 stop csci567-sam` from WSL. The alternative `--leave-active` explicitly disables automatic shelving and requires someone to stop billing manually; it is unsuitable for an unattended overnight run.

Transfer the existing authoritative `gpu_budget.json` and verified `cifar-10-binary.tar.gz` into `/workspace/sam_inputs/` using your secure file-transfer method. These are inputs for migrating this experiment, not a whole-tree results upload. Do not put either in Git or start a zero ledger. Local handoff: `C:/Users/Siddharth Magesh/AppData/Local/Temp/sam_main_machine_handoff/`. The carried ledger has 0.8682869146 worker GPU-hours charged; Jetstream allocated idle/setup time is additional SU usage and is not represented by that ledger.

## 6. First run: setup and preflight only

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
bash experiments/start_jetstream.sh preflight --execute --setup \
  --ledger-import /workspace/sam_inputs/gpu_budget.json \
  --cifar-archive /workspace/sam_inputs/cifar-10-binary.tar.gz \
  --selection all --max-hours 0.75 \
  --openstack-bin /workspace/js2-control/bin/openstack
```

The 45-minute launcher limit includes setup; at the guide's full-A100 rate of 64 SUs/hour this is approximately 48 SUs plus start/shelve latency, not an exact billing cap. The independent worker supervisor also enforces the cumulative 40-hour experiment limit. The launcher requests shelving on success, failure or handled interruption, after workers exit. Cloud/API errors, abrupt VM/process death or a teammate's running GPU job can prevent automatic shelving. Check the instance's final SHELVED_OFFLOADED state in WSL/Exosphere; a Linux shutdown is not a substitute. Never shelve over another active job. Do not rely on the staff's two-hour idle cleanup.

The launcher prints `SESSION=/workspace/sam_paper_replication/runs/...` to its log. Preserve this exact folder. Both strict preflights must finish successfully. The T4 equivalence warning does not bypass the A100 gate. If either fails, inspect the log; do not start training. Use a fresh session after a code change while keeping the same ledger.

## 7. Full runs only after preflight passes and costs fit

Review the selected-run cost estimate and remaining allocation in Exosphere. The entire nine-run matrix remains SAM-200, SGD-200 and SGD-400 for seeds 0, 1, 2. Do not shrink epochs or drop seeds to claim completion. Start the same instance again if it auto-shelved, reconnect, and substitute the exact SESSION below:

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
bash experiments/start_jetstream.sh train --execute \
  --session /workspace/sam_paper_replication/runs/REPLACE_WITH_SESSION \
  --selection all --max-hours "${APPROVED_HOURS:?Set the hours approved after preflight}" \
  --openstack-bin /workspace/js2-control/bin/openstack
```

Set `APPROVED_HOURS` to the wall-time limit justified by preflight and your allocation; the command refuses to run until it is set. This is a limit, not a runtime estimate. At 64 SUs/hour, 39 allocated hours would be 2,496 SUs. The selected estimated remaining work must fit the remaining 40-hour worker ledger. For staged execution, select one exact label such as `SAM-200_seed_0`. A deadline can interrupt an epoch; recovery restores the last fully committed epoch. If a run was interrupted, rerun with the same SESSION/selection plus `--resume`; completed runs are skipped after provenance checks. Do not change the notebook worker between preflight and training.

Training completion renders the notebook's final-error plots and paired statistics into `SESSION/summary/`. Missing runs are reported as incomplete, not aggregated. To render saved/partial results separately, use the `report` phase with the same `--session` and `--execute` (and normal shelving credentials).

## 8. Monitor, disconnect and retrieve

```bash
# On csci567-sam:
tmux attach -t sam-wrn
# Ctrl-b, then d detaches. Closing the laptop does not end tmux.
tail -f /workspace/sam_paper_replication/launcher_logs/YOUR_LOG.log
```

A finished tmux session disappears; its log and `launcher_*.json` receipt remain. For a graceful stop, attach and press Ctrl-c; allow worker cleanup and shelving to finish. Do not kill the supervisor or reset its ledger reservation. Jobs survive SSH loss, not a VM shelf/reboot/deletion. Results remain on the persistent volume after shelving. Reconnect later to download needed results, then run `~/bin/js2 stop csci567-sam` from WSL if still active. The SU allocation and the experiment's worker-time ledger are different budgets; keep both within limits.


## 9. Results and interpretation

The plots and tables in `SESSION/summary/` include training-progress test-error curves, final per-seed errors, all four displayed baselines (SAM-200, SGD-200, SGD-400, paper-style selected SGD), and paired SGD-minus-SAM differences with mean, sample SD and Student-t 95% CI. Training metrics measured at SAM's perturbed weights are not a fair direct comparison with SGD training loss. Use final-epoch official test results.

The predeclared reproduction rule requires all nine runs complete, lower mean SAM error, a paired paper-style SGD-minus-SAM 95% CI entirely above zero, and means within 0.3 percentage points of SAM 2.7% and SGD 3.5%. Paper-style SGD means the lower final error of the independently trained 200- and 400-epoch SGD runs for each seed. It never selects an intermediate epoch. Incomplete runs remain incomplete; do not change the rule after observing results.

Do not use the old ResNet-18 pilots or the ten-batch smoke accuracy as a reproduction of this WRN result. A100 outcomes remain (to verify) until actual logs exist.

## 10. Recovery and troubleshooting

| Message or situation | Action |
|---|---|
| Full A100 requirement fails | Check `nvidia-smi`; a 10 GB or 20 GB slice is not the approved target. Do not shrink the model/batch or weaken the guard silently. |
| Metadata UUID mismatch | You are on a different/rebuilt instance or metadata access failed. Confirm ownership and identity; do not disable the guard. |
| tmux `sam-wrn` already exists | Inspect that session and logs; do not launch a duplicate or kill a teammate's job. |
| Missing cloud CLI/credentials | Complete the separate control setup; the instance may still bill if this check fails before cleanup is armed. Shelve manually from WSL when idle. |
| Ledger missing or divergent | Transfer/reconcile the authoritative ledger. Never create a zero ledger to bypass the cap. |
| Ledger retains an unresolved reservation | A supervisor may have died. Reconcile against scheduler/process evidence; do not manually zero it. |
| Numerical preflight fails | Keep its diagnostics and logs. Do not relax tolerances to force a pass or proceed to training. |
| CUDA/package failure | Setup keeps the pinned CUDA 11.0 runtime and installs NVIDIA label `cuda-11.3.1`'s `cuda-nvcc=11.3.122` solely for its CUDA 11.x compiler. The worker prepends the runtime `lib` directory and a validated CUDA 11.x `ptxas`, sets `CUDA_HOME`/`CUDA_PATH`, and runs a minimal JAX GPU JIT before preflight. Inspect those setup records; retain legacy pins. Do not install modern JAX/Flax or use a different implementation silently. |
| GPU out of memory | Record the failure and verify actual GPU memory. A100 feasibility is still (to verify). |
| Interrupted training | Keep exact SESSION/config/worker/environment; rerun selected run with `--resume`. A partial epoch is rerun and charged. |
| Session source/config mismatch | Use a new session with the same ledger after reviewing the change; do not overwrite provenance. |
| Deadline reached | Inspect latest committed checkpoint and ledger, then resume only within approved remaining budgets. |
| No tmux session after launch | It may have exited quickly or auto-shelved. Read the log and launcher receipt; do not assume success. |
| Shelving API error | From WSL run `~/bin/js2 stop csci567-sam` once no job is active, then verify SHELVED_OFFLOADED. |

The runner requires all selected work to fit the experiment ledger; the separate `--max-hours` deadline may interrupt an otherwise valid selection. Choose its duration using measured preflight timings and approved SU spend. A 45-minute preflight and subsequent training are separate invocations with the same ledger and SESSION.

## 11. Copy this prompt into another chat

```text
Read docs/jetstream_runbook.md, docs/project_coding_guidelines.md,
docs/notes.md and docs/project_tracking.md before acting.

We are replicating the SAM paper's WRN-28-10/CIFAR-10 Table 1 row using
original pinned Python 3.8/JAX 0.2.6/jaxlib 0.1.57+cuda110/Flax 0.2.2 code.
The full design is SAM-200, SGD-200, SGD-400; seeds 0,1,2; batch 256 split
into eight logical batches of 32. Do not reimplement SAM, change the
hyperparameters/seeds, or silently upgrade the dependencies.

Only use csci567-sam (recorded UUID in the runbook). Do not touch any other
machine, create/delete instances, or resize without explicit approval.
The last inspected GPU was a 10 GB slice; the runner requires a full 40 GB
A100. Check actual capacity and instance identity first. Keep everything
on the mounted /workspace volume and use the tracked tmux launcher.

The T4 smoke was incomplete: SGD passed functional checks, SAM ran out of
memory, and original-step repeatability made numerical equivalence
inconclusive. Full A100 preflight and automatic shelving are unverified.
Do not describe this as a successful reproduction or bypass strict gates.

First verify inputs, the existing authoritative ledger, cloud-control
credentials and the requested phase. Preflight only comes first. Full
training needs passed preflight plus measured runtime and sufficient
approved SUs. Keep the cumulative 40-worker-hour budget and separate
allocation charges. Preserve failed runs, logs and checkpoints.

The launcher requests shelving after success/failure; verify the actual
SHELVED_OFFLOADED status and never shelve over another running job. Do not
leave an idle GPU ACTIVE. Do not ask for or paste credentials into chat.

Start by reporting the current machine state, readiness and intended
bounded command. Do not start a long run until its budget is approved.
```
