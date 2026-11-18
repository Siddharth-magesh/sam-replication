# SAM replication on Jetstream2: runbook and chat handoff

## Current first comparison: follow run_config.md

**Latest state:** `20261118_072652` again failed the numerical gate; `--leave-active` correctly kept SSH open. Do not repeat the full comparison while that issue is unresolved. The diagnostic has now been tightened to warm both transforms before measurement, reuse the same prepared reference inputs across calls, and verify input immutability. This tests a possible harness/first-execution effect; it is not a claimed resolution or a tolerance change.

For one bounded, attended **SGD diagnostic only**, on the instance:

```bash
cd /workspace/sam-replication
git switch repro
git pull --ff-only origin repro
python3 -u experiments/jetstream_wrn.py preflight --execute \
  --diagnostic-only --selection SGD-200_seed_0 \
  --deterministic-cudnn --leave-active --max-hours 0.1
```

This uses the installed environment/cache, has a six-minute invocation cap, runs neither an epoch nor main training, and leaves the machine connected. The diagnostic-only dispatcher now runs exactly the selected method. Print the new SESSION's `preflight/SGD-200_seed_0/diagnostics.json`; the report records `reference_protocol=warmed_same_prepared_inputs_v1` and `input_integrity`. A completed diagnostic-only process is not a passed preflight. Keep earlier reports; no old failure is overwritten. If idle afterward, manually shelve as below.

### Current debugging mode: foreground Python, keep the instance connected

The latest supplied `20261118_071943` run still failed the original-reference repeatability gate with `TF_CUDNN_DETERMINISTIC=1`. Main training has not started. The subsequent SSH disconnection was the launcher's deliberate automatic shelving, not evidence that tmux killed the worker. Do not reinstall packages or repeatedly retry the unchanged numerical check.

For a future attended attempt, use the existing Python entry point directly with `--leave-active`, instead of the detached shell launcher. This keeps the VM active on success, failure or handled interruption and does not require cloud-control credentials for shelving:

```bash
cd /workspace/sam-replication
python3 -u experiments/cifar10_first_comparison/run.py comparison \
  --execute --leave-active --deterministic-cudnn \
  --max-hours 24 --record-l4-smoke
```

This is the foreground launch syntax, **not a fix for the outstanding numerical failure**. Do not start a duplicate while another worker is running. The 24-hour invocation and cumulative worker-time limits remain; `--leave-active` disables cloud shelving only. A failed Python command returns to your SSH prompt; saved logs and diagnostic JSON remain in the printed SESSION. A foreground run is tied to the SSH terminal: keep it connected. To make that same command survive laptop disconnection, enter `tmux new -s sam-debug` first, run it there, and detach with Ctrl-b then d; the shell remains available after Python exits.

An active/idle VM continues consuming allocation even after Python exits; the worker ledger does not include that idle time. When no work remains, run from WSL:

```bash
js2 stop csci567-sam
js2 status csci567-sam
```

Confirm SHELVED_OFFLOADED. The unattended single-job workflow below still auto-shelves unless `--leave-active` is explicitly supplied; reserve it for after the numerical failure has been resolved.

**The single-job commands below are for after the numerical issue is resolved. Use the bounded diagnostic above for now.** Steps A–F are the alternative staged/reference workflow. Commands labelled WSL run on your laptop; commands labelled instance run after connecting to csci567-sam. Do not run the later nine-run examples for this first comparison.

### Installed instance: preflight → SAM-200 → SGD-200 in one job

**Recorded runtime option: `--deterministic-cudnn` did not resolve the failure in subsequent supplied runs.** Saved diagnostic `20261118_070302` shows identical inputs but a 5.55% optimizer-state relative difference between two original reference calls (wrapper/reference: 5.49%). Numerical equivalence remains inconclusive; do not reuse that failed session. The option sets `TF_CUDNN_DETERMINISTIC=1` for preflight and both training methods, records it in CONFIG/manifest, and rejects a worker whose environment differs. It does not relax numerical or BatchNorm gates, alter SAM/SGD hyperparameters, or install anything. Use a fresh session and retain the same option on resume/report. Actual A100 resolution is **(to verify)**.

Source verification: [jaxlib 0.1.57 WORKSPACE](https://github.com/jax-ml/jax/blob/jaxlib-v0.1.57/WORKSPACE) pins TensorFlow/XLA commit `4cf039a9877a5e7a73cf3bc189af0bd03d2cb422`. Its [cuDNN backend](https://github.com/tensorflow/tensorflow/blob/4cf039a9877a5e7a73cf3bc189af0bd03d2cb422/tensorflow/stream_executor/cuda/cuda_dnn.cc) reads this variable and excludes nondeterministic backward algorithms; the [XLA convolution picker](https://github.com/tensorflow/tensorflow/blob/4cf039a9877a5e7a73cf3bc189af0bd03d2cb422/tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.cc) also honors it. This is a supported legacy runtime control, not an assumed modern XLA flag. Record this execution-setting deviation in results; it can change performance and numerics.

The November 18 UTC preflight log exposed a bug in our diagnostic wrapper: the reference pmap received an unreplicated optimizer, including a scalar step. The corrected wrapper adds the one-device axis to every reference optimizer leaf and unwraps the reference learning-rate axis before comparison. Original SAM/SGD training logic, BatchNorm behavior and numerical tolerances are unchanged. This is a fresh worker revision: do not reuse the failed `20261118_064611` session.

On **WSL**, if the machine has auto-shelved, start only this machine, then connect. If already ACTIVE, skip `start`:

```bash
~/bin/js2 start csci567-sam
~/bin/js2 ssh csci567-sam
```

On the **instance**, inspect the checkout first. If `git status --short` prints changes, preserve/review them before switching or pulling:

```bash
cd /workspace/sam-replication
git status --short
git switch repro
git pull --ff-only origin repro
export OS_CLOUD=CIS262089_IU
bash experiments/cifar10_first_comparison/start.sh comparison --execute --deterministic-cudnn \
  --max-hours 24 --record-l4-smoke
```

This command reuses `/workspace/sam_paper_replication/envs/sam`, its pinned runtime checkout, the persistent CIFAR cache and the existing authoritative `gpu_budget.json`. **Do not add `--setup`, `--ledger-import` or `--cifar-archive` for this installed machine.** Missing local prerequisites are checked before tmux starts. `--setup` is reserved for a genuinely absent/broken environment; it is not required on each start/unshelve.

`comparison` creates and prints a new SESSION, runs both strict preflights once, sums the actual timing estimates for the two selected runs and starts SAM-200 seed 0 followed by SGD-200 seed 0 only if the gates and budget pass. It renders results and requests shelving at the end (or on failure); there is no shelf/restart between phases. Twenty-four hours is a maximum invocation duration, not an estimate; at the guide's 64 SU/h rate it caps planned allocation at approximately 1,536 SUs plus lifecycle latency. Check remaining course allocation before launching. The independent cumulative 40-worker-hour limit still applies.

`--record-l4-smoke` explicitly adds the user-reported 0.32108525306166674-hour completed L4 smoke once, under a stable attempt ID, preserving existing ledger entries and writing `gpu_budget.before_l4_import.json` before replacement. Repeating the option does not double-charge. It excludes separate initialization attempts whose timings were not saved; accounting for those remains **(to verify)**. No missing time is silently declared zero.

Monitor on the **instance**:

```bash
tail -f /workspace/sam_paper_replication/launcher_logs/latest_first_comparison.log
```

Ctrl-c exits `tail` only. The detached tmux job continues after SSH/laptop disconnects. To interact with the job itself, use `tmux attach -t sam-wrn`; Ctrl-b then d detaches. The latest log contains the exact SESSION path and any failure; do not repeatedly reinstall or launch another job if the numerical gate fails. Corrected A100 preflight remains **(to verify)** until an actual pass.

If interrupted **after preflight passed**, retain the printed SESSION and use the same code revision:

```bash
export SESSION=/workspace/sam_paper_replication/runs/REPLACE_WITH_PRINTED_SESSION
bash experiments/cifar10_first_comparison/start.sh comparison --execute --deterministic-cudnn \
  --session "$SESSION" --resume --max-hours 24 --record-l4-smoke
```

Completed preflight/methods are skipped after provenance checks, incomplete training resumes from its committed checkpoint, and an unstarted method starts fresh. An unsuccessful preflight needs investigation; it is not converted into a pass. Use step F to verify shelving and inspect final results.

Latest evidence (November 17, 2026 local time): the user-run Colab L4 test reported **both SGD and SAM functional smoke tests passed**, ten batches each at batch 256 / eight sub-batches. It checked finite losses/state, parameter and BatchNorm updates, evaluation, checkpoint restoration and shared initial weights. Recorded time was **0.32108525306166674 hours (19.27 minutes)**, including data preparation and compilation. SGD's 13.56 minutes versus SAM's 5.70 minutes is not a speed comparison: SGD downloaded/prepared CIFAR first. A100 strict numerical preflight and unattended shelving remain **(to verify)**.

The current user-selected scope is **SAM-200 seed 0, then SGD-200 seed 0**. No extra seeds or SGD-400 yet. This section supersedes the full-nine-run launch examples below; that larger experiment remains available for later work. Use branch **repro**. No Jetstream instance was started for this packaging change.

Only four new files are needed in `experiments/cifar10_first_comparison/`:

| File | Role |
|---|---|
| `config.json` | Fixed experiment and environment manifest; validated against notebook configuration and original pins. It is not a hyperparameter override file. |
| `run.py` | Selects preflight, seed-0 SAM, seed-0 SGD or report through the existing guarded runner. Plan-only unless `--execute` is supplied. |
| `start.sh` | Validates arguments then launches `run.py` in detached tmux, with persistent logs. Plan-only without `--execute`. |
| `report.py` | Reads both completed 200-epoch records and prints test errors, gaps from paper numbers and the paired difference. No GPU, no file writes. |

Clone the **whole repository**, because these entry points reuse `experiments/jetstream_wrn.py` and notebook 04. The original SAM training step, eight logical sub-batches, separate BatchNorm states, strict numerical preflight, recovery and budget ledger remain unchanged. The plain `python -m sam.sam_jax.train ...` command supplied in the request has the correct main hyperparameters, but it uses the number of physical JAX devices: on one GPU it gives one batch of 256, not eight sub-batches of 32. It also expects a source directory named `sam`, importable from its parent, and appends hyperparameter subdirectories to output_dir. Use the wrapper for this single-GPU experiment.

Environment: no Windows venv is created. `preflight --setup` creates or repairs `/workspace/sam_paper_replication/envs/sam` with Python 3.8 using micromamba, including CUDA 11.0/cuDNN 8.0.5 and the archived CUDA 11.3 compiler. It installs the original requirements with the existing documented runtime-only absl-py 0.10.0 adjustment and GPU jaxlib 0.1.57+cuda110. JAX 0.2.6 / Flax 0.2.2 stay pinned. Original requirements and authors' Python sources are untouched. Do not run the guide's generic `js2 setup` (its modern vLLM stack is unrelated). The setup runs every time explicitly requested, repairing partially installed environments rather than trusting that an interpreter exists.

Before execution, complete the existing storage, cloud-control credential and authoritative-ledger prerequisites in sections 4–5 below. Confirm **csci567-sam** and a full A100 40 GB; a 10 GB slice is rejected. No script resizes or creates a VM. Ensure `OS_CLOUD` identifies your existing credential configuration. Cloud credentials stay outside Git. Import the latest authoritative ledger, including any later Jetstream attempts; do not replace it with an older Colab copy or a fresh zero ledger.

### A. Connect to the right machine — WSL terminal

Check the instance and course allocation before allocating GPU time. Only this instance is authorized. The guide quotes 64 SUs/hour for the required full A100: the first 45-minute invocation is approximately 48 SUs plus startup/shelving time. A 10 GB slice will be rejected; do not resize or substitute another machine through these instructions.

```bash
~/bin/js2 status csci567-sam
```

If SHELVED, and you are ready to run the bounded preflight, start it. Skip the start command if already ACTIVE:

```bash
~/bin/js2 start csci567-sam
~/bin/js2 gpus csci567-sam
~/bin/js2 ssh csci567-sam
```

### B. Clone and select repro — instance terminal

For a **new checkout**:

```bash
mountpoint /workspace
nvidia-smi
# Stop here if /workspace is not mounted or the GPU is not a full A100 40 GB.
cd /workspace
git clone --branch repro https://github.com/Siddharth-magesh/sam-replication.git sam-replication
cd sam-replication
git switch repro
git branch --show-current
git log -1 --oneline
```

For an **existing checkout**, use this instead of cloning again:

```bash
cd /workspace/sam-replication
git status --short
```

If this prints local changes, preserve/review them before continuing; do not reset or overwrite them. When clean:

```bash
git fetch origin
git switch repro
git pull --ff-only origin repro
git branch --show-current
git log -1 --oneline
```

The branch output must be `repro`. Keep this exact code revision throughout preflight and training. Do not pull updates midway through a session.

### C. Prepare cloud control and inputs — instance terminal

Complete the credential setup in section 5 before running a GPU job. It creates the separate `/workspace/js2-control` environment; the SAM environment is created later by `--setup`.

```bash
mkdir -p /workspace/sam_inputs
export OS_CLOUD=CIS262089_IU  # Use the cloud name in your existing credentials.
/workspace/js2-control/bin/openstack server show \
  195e6a8f-8ae1-45be-bcd7-7845b0b2419f -c name -c status
```

Confirm that the response names `csci567-sam`. Securely transfer the authoritative `gpu_budget.json` and optional verified `cifar-10-binary.tar.gz` into `/workspace/sam_inputs/`. A prepared TFDS cache can instead be copied to `/workspace/sam_paper_replication/tensorflow_datasets/`, preserving its directory structure. Do not upload the Colab Python environment.

**Budget prerequisite:** the standalone L4 `attempts.json` is evidence, not a replacement for the authoritative supervisor ledger. The single-job command above uses `--record-l4-smoke` to charge its known total once. Preserve/account for other unrecorded diagnostic attempts separately. Do not import the old 0.8683-hour Colab handoff as though it includes later work. The existing Jetstream ledger is authoritative; its latest supplied dump already includes the earlier Jetstream attempts.

### D. Setup and preflight only — instance terminal

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
python3 experiments/cifar10_first_comparison/run.py preflight  # safe plan-only check
bash experiments/cifar10_first_comparison/start.sh preflight --execute --deterministic-cudnn \
  --max-hours 0.75
```

The archive argument is optional if the existing TFDS cache is complete or normal downloading works. The authoritative ledger is mandatory on migration; subsequent invocations reuse it. The default separate OpenStack CLI path is `/workspace/js2-control/bin/openstack`. Preflight runs both methods for timing and correctness, then requests shelving. Its 45-minute invocation cap includes setup; at the guide's 64 SU/h rate, plan approximately 48 SUs plus shelving/start latency. This is not a guarantee that setup/preflight will finish in 45 minutes.

Copy the printed `SESSION=...` path. Both preflights must pass. Read the measured per-method estimates, the remaining worker budget and the live course allocation. For the first comparison add only the SAM-200 and SGD-200 estimates, not nine runs. Each training invocation independently checks its selected remaining work against the shared ledger. Do not bypass a failed numerical gate or treat the old 5–11 / 3–6 hour guesses as measurements.

### E. Run SAM, then SGD — only after preflight passes

After reviewing the measured costs and starting the same instance again if shelved, reconnect using step A. Set `APPROVED_HOURS` to the measured, reviewed wall-time limit for the specific method (for example, enter `export APPROVED_HOURS=...` with your actual reviewed number; do not type the dots). Then, on the instance:

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
export SESSION=/workspace/sam_paper_replication/runs/REPLACE_WITH_PREFLIGHT_SESSION
# Set APPROVED_HOURS to the reviewed invocation limit; no long run is enabled by default.
bash experiments/cifar10_first_comparison/start.sh sam --execute --deterministic-cudnn \
  --session "$SESSION" --max-hours "${APPROVED_HOURS:?Set a reviewed SAM limit first}"
```

Wait for SAM completion; the runner requests shelving. Restart **only csci567-sam**, reconnect, restore SESSION and OS_CLOUD, and set the reviewed SGD limit before:

```bash
bash experiments/cifar10_first_comparison/start.sh sgd --execute --deterministic-cudnn \
  --session "$SESSION" --max-hours "${APPROVED_HOURS:?Set a reviewed SGD limit first}"
```

SGD refuses to start until SAM's status is complete in that session. Add `--resume` to the same SAM/SGD command after interruption; keep the same session, code, environment and authoritative ledger. The last fully committed epoch is restored. Each command is a separate job; SGD is not automatically launched after shelving. `tmux attach -t sam-wrn` shows output; Ctrl-b then d detaches. Closing the laptop does not stop tmux. Inspect `/workspace/sam_paper_replication/launcher_logs/first_comparison_*.log` and the launcher receipt. Pre-launch failures can occur before auto-shelving is armed: check the log promptly and verify SHELVED_OFFLOADED from WSL/Exosphere after completion/failure. Do not assume Linux shutdown stops billing.

### F. Inspect results and verify shelving

Each successful training phase generates the existing `SESSION/summary/test_error_curves.png` and `final_per_seed.csv`. After **both** runs finish, this read-only command prints the compact comparison (also works on downloaded results with system Python):

```bash
python3 experiments/cifar10_first_comparison/report.py "$SESSION"
```

From **WSL**, verify the final machine state:

```bash
~/bin/js2 status csci567-sam
```

If still ACTIVE and your job has ended with no other work running, release it:

```bash
~/bin/js2 stop csci567-sam
~/bin/js2 status csci567-sam
```

Confirm SHELVED_OFFLOADED. Do not delete the instance or volume. Results persist on `/workspace`; retrieve them when connected and preserve the authoritative ledger with them.

Report epoch-200 error, the gap from SAM 2.7% / SGD 3.5%, and SGD-200 minus SAM in percentage points. One seed provides no SD/CI, and paper SGD is the better of 200/400 epochs. The notebook's full reproduction assessment correctly remains incomplete with only two of nine planned runs; this is not a failed first-comparison job or a claim of full reproduction.

Validation for this package: Python 3.8 syntax, JSON/configuration and no-execution CLI checks passed locally. The subsequent user-run L4 functional smoke passed as described above. Full A100 preflight, numerical equivalence, A100 memory fit and unattended cloud lifecycle remain **(to verify)**. Earlier T4 SAM OOM and Jetstream initialization failures remain recorded; an L4 pass does not resolve the A100-specific findings.

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

- The historical T4 smoke did **not** pass. SGD passed ten updates, finite losses, parameter changes, BatchNorm checks, evaluation and checkpoint restoration; SAM exhausted T4 memory. The Jetstream `smoke` phase is a separate one-epoch functional test and does not replace strict preflight.
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

Transfer the existing authoritative `gpu_budget.json` and verified `cifar-10-binary.tar.gz` into `/workspace/sam_inputs/` using your secure file-transfer method. These are inputs for migrating this experiment, not a whole-tree results upload. Do not put either in Git or start a zero ledger. Local handoff: `C:/Users/Siddharth Magesh/AppData/Local/Temp/sam_main_machine_handoff/`. That historical handoff ledger has 0.8682869146 worker GPU-hours charged and excludes the later L4 smoke and subsequent attempts; reconcile it before use. Jetstream allocated idle/setup time is additional SU usage and is not represented by that ledger.

## 6. First run: setup and preflight only

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
bash experiments/start_jetstream.sh preflight --execute --deterministic-cudnn --setup \
  --ledger-import /workspace/sam_inputs/gpu_budget.json \
  --cifar-archive /workspace/sam_inputs/cifar-10-binary.tar.gz \
  --selection all --max-hours 0.75 \
  --openstack-bin /workspace/js2-control/bin/openstack
```

The 45-minute launcher limit includes setup; at the guide's full-A100 rate of 64 SUs/hour this is approximately 48 SUs plus start/shelve latency, not an exact billing cap. The independent worker supervisor also enforces the cumulative 40-hour experiment limit. The launcher requests shelving on success, failure or handled interruption, after workers exit. Cloud/API errors, abrupt VM/process death or a teammate's running GPU job can prevent automatic shelving. Check the instance's final SHELVED_OFFLOADED state in WSL/Exosphere; a Linux shutdown is not a substitute. Never shelve over another active job. Do not rely on the staff's two-hour idle cleanup.

The launcher prints `SESSION=/workspace/sam_paper_replication/runs/...` to its log. Preserve this exact folder. Both strict preflights must finish successfully. The T4 equivalence warning does not bypass the A100 gate. If either fails, inspect the log; do not start training. Use a fresh session after a code change while keeping the same ledger.

For a bounded one-epoch functional smoke on the full A100, use the separate `smoke` phase. It runs fresh seed-0 SGD-200 and SAM-200, preserves the original eight logical sub-batches and optimizer step, records finite losses/updates, epoch time, sampled GPU memory, and checkpoint restoration, and never creates a preflight pass:

```bash
export OS_CLOUD=CIS262089_IU
bash experiments/start_jetstream.sh smoke --execute --deterministic-cudnn \
  --selection all --max-hours 0.5 \
  --openstack-bin /workspace/js2-control/bin/openstack
```

Inspect both `smoke/SGD-200_seed_0/smoke.json` and `smoke/SAM-200_seed_0/smoke.json`. A complete smoke is functional evidence only; it does not authorize `train`, bypass numerical verification, or certify the reproduction. It uses the same authoritative ledger and automatic shelving.

For a bounded one-replica numerical harness diagnostic after a code change, use one selected method and stop before the eight-replica/full-epoch checks:

```bash
bash experiments/start_jetstream.sh preflight --execute --deterministic-cudnn --diagnostic-only \
  --selection SGD-200_seed_0 --max-hours 0.25 \
  --openstack-bin /workspace/js2-control/bin/openstack
```

Inspect that session's `preflight/SGD-200_seed_0/diagnostics.json`. Only run the strict preflight after the diagnostic reports repeatable original `pmap` output and equivalent `vmap` output.

## 7. Full runs only after preflight passes and costs fit

Review the selected-run cost estimate and remaining allocation in Exosphere. The entire nine-run matrix remains SAM-200, SGD-200 and SGD-400 for seeds 0, 1, 2. Do not shrink epochs or drop seeds to claim completion. Start the same instance again if it auto-shelved, reconnect, and substitute the exact SESSION below:

```bash
cd /workspace/sam-replication
export OS_CLOUD=CIS262089_IU
bash experiments/start_jetstream.sh train --execute --deterministic-cudnn \
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
| `CUDA_ERROR_ILLEGAL_ADDRESS` during smoke initialization | Preserve the failed session and log. The smoke worker avoids a redundant second WRN initialization; retry only after pulling the corresponding commit, and treat any recurrence as a pinned JAX/driver compatibility issue requiring separate investigation. |
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
