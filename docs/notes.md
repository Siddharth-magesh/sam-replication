## L4 functional smoke result (November 17, 2026 local time)

User-supplied output reports both ten-batch SGD and SAM smoke tests passed on Colab L4 using the pinned original implementation and eight sub-batches of 32. Model initialization reported 36,220,634 finite parameters. Result folder: `smoke_20261118_043333_653758` (UTC name). The checks covered finite metrics/state, parameter and BatchNorm updates, evaluation, checkpoint restoration and identical initial-weight hashes. Recorded worker time was 0.32108525306166674 hours; SGD 13.56 minutes included initial CIFAR download/preparation, SAM 5.70 minutes reused it. These are not comparative training-speed measurements. Actual raw result files and backup completeness remain (to verify); the outcome is recorded from user-provided console output.

The standalone diagnostic needed two harness fixes: check jaxlib's CUDA suffix using package metadata (module __version__ is 0.1.57), and parse Abseil flags before model initialization. The main notebook already implements the compatible module check and flag parsing; no training-code changes were needed. L4 CUDA compiler warnings fell back to the driver and did not prevent functional success. Strict A100 preflight, numerical equivalence and shelving remain (to verify). The standalone attempts.json has not yet been reconciled into the authoritative 40-hour ledger; preserve earlier failed initialization attempts as well.

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


## WRN-28-10 T4 smoke execution (2026-11-16 local / November 17 UTC)

Executed the pinned Python 3.8 / JAX 0.2.6 / jaxlib 0.1.57+cuda110 / Flax 0.2.2 worker on the existing Colab CLI T4 session. Original SAM/SGD code and model/hyperparameters were unchanged. SGD passed ten batch-256 updates with eight logical sub-batches, finite losses, changed parameters, independent first-pass BN states, evaluation over 320 examples and exact checkpoint tensor restoration at steps 5 and 10. This is a functional smoke, not a paper accuracy result. SAM failed with GPU out-of-memory at the eight-replica diagnostic, requesting a 6.16 GiB allocation. Releasing discarded reference/candidate buffers earlier did not resolve it. Combined smoke: FAILED / INCOMPLETE.

Fixed checkpoint metadata serialization: Flax 0.2.2/msgpack rejects numpy.int32 scalars, so completed_epoch now uses a Python int in the payload and restore template. Both notebooks release discarded diagnostic outputs before the eight-lane step. Notebook outputs remain cleared; original sam_jax files and requirement pins remain untouched.

The one-replica numerical comparison was not a reliable standalone equivalence test on this T4: a repeat of the unchanged original pmap call on identical inputs differed by optimizer relative L2 0.006275, comparable to vmap differences. Disabling convolution autotuning did not solve the strict one-replica check; that experimental flag was NOT adopted. The smoke now records reference repeatability separately and permits functional checks only when differences remain within a declared three-times-reference-noise bound (floor 1e-3). Such a result explicitly remains inconclusive, not numerical equivalence. Main replication preflight retains the fixed strict gate and additionally requires the reference to be repeatable. A100 equivalence and full replication remain (to verify).

The slow official CIFAR archive was transferred in ranges through WSL and reassembled into TFDS's supported manual cache. Size 170052171 bytes and SHA256 c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd match pinned TFDS checksums. No data or augmentation changes. The interrupted download, failed diagnostics and successful SGD attempt remain charged: authoritative total 0.8682869146130557 worker GPU-hours, leaving 39.13171308538694 of 40. Installation and idle allocation time are outside this worker ledger. No full training was launched. A proposed further SAM workspace probe was NOT executed because CLI credential refresh was unavailable; the user chose main-machine instructions instead.


## Jetstream background runner (2026-11-17)

Current packaging update: `run_config.md` now selects SAM-200 seed 0 followed by SGD-200 seed 0. `experiments/cifar10_first_comparison/` provides a checked manifest, plan-only Python selector, tmux launcher and read-only paired report. It delegates execution to the existing runner/notebook rather than copying or changing the authors' SAM implementation. Environment bootstrap remains the existing Python 3.8 micromamba setup on `/workspace`, with no local Windows environment installation. The full notebook matrix and all executed notebook outputs are unchanged by this update.

The supplied plain train command is a valid upstream entry point when `sam` is importable from its parent, but it partitions batches by physical devices (`sam_jax/train.py`), not eight virtual sub-batches on one GPU. The selected wrapper is needed for m=32 on one A100. The parameter-count assertion against a second factory-built model was removed by earlier initialization debugging; current `architecture_check.json` records the configured model/count and is not independent factory-equivalence evidence **(to verify)**. This packaging does not bypass the numerical preflight or resolve the earlier illegal-address/OOM failures.

Validation: Python 3.8 AST syntax, shell syntax, notebook JSON and empty outputs, config/pin checks and all four plan-only CLI phases passed. Full notebook schema validation was not available in the local Python environment (`nbformat` absent), so schema validation is **(to verify)**; no package was installed for this. Colab CLI testing is **(to verify)**: the WSL tool request was rejected before access/authentication could be checked, and was not retried. No new GPU worker ran; live budget balance and GPU readiness remain **(to verify)**.

`experiments/jetstream_wrn.py` reads the reviewed worker, supervisor, budget gates and plotting code from `repro_notebooks/04_wrn28_10_cifar10_paper_replication.ipynb`. The pinned bootstrap now lives in the same WRN replication notebook. The separate Colab smoke notebook has been removed; keep the WRN notebook in the clone. No SAM/SGD reimplementation. Execution requires `--execute`; otherwise the command only prints a plan. `start_jetstream.sh` starts a detached tmux session named sam-wrn; logs, environment, dataset, checkpoints and reports stay on the persistent `/workspace` volume. The main notebook remains disabled by default.

The pinned setup keeps the CUDA 11.0 runtime and installs NVIDIA label `cuda-11.3.1`'s `cuda-nvcc=11.3.122` solely for its CUDA 11.x compiler because the runtime package does not include `ptxas`. It prepends the CUDA runtime `lib` directory, selects a working CUDA 11.x `ptxas`, sets `CUDA_HOME` and `CUDA_PATH`, and passes the same environment to workers. It validates `ptxas --version` and runs a minimal JAX GPU JIT before preflight, preventing a system CUDA 13.x `ptxas` from being selected by XLA. TensorFlow's legacy CUDA 10.x library warnings remain separate because TensorFlow GPU devices are hidden.

The one-replica verifier now snapshots independent copies of optimizer, BatchNorm state, inputs and RNG keys, synchronizes two original `pmap` calls back-to-back before compiling the alternate `vmap`, and records parameter, optimizer, BatchNorm, metric and learning-rate differences with the first divergent leaf. This targets verification-harness sequencing/aliasing without changing the original training step; A100 numerical validation remains pending.

Only server csci567-sam, UUID 195e6a8f-8ae1-45be-bcd7-7845b0b2419f, is accepted, verified using the instance metadata service. The last inspected flavor was g3.medium (10 GB A100 slice); the runner requires a full 40 GB A100 because the unchanged eight-replica SAM already exhausted T4 memory. Do not delete/recreate or resize another team's instance. Arrange the hardware change separately; partial-to-full GPU driver compatibility is (to verify). SSH access, full A100 numerical preflight, runtime, tmux execution and automatic shelving with this runner remain (to verify). Python 3.8 syntax, shell syntax, notebook selectors and plan-only CLI checks passed. These checks are not GPU validation.

The Jetstream `smoke` phase is a separate bounded functional test. It runs one complete 195-batch epoch for fresh seed-0 SGD-200 and SAM-200 using the original mapped training step, eight logical sub-batches, and the fixed 200-epoch schedule; it records finite losses, parameter changes, epoch runtime, sampled `nvidia-smi` memory, and save/restore evaluation. It does not run or satisfy the strict numerical preflight, does not produce a preflight pass, and does not authorize full training. Smoke time remains charged to the same cumulative ledger and the normal instance identity and auto-shelving guards remain active.

The first Jetstream smoke attempt (`20261117_174647`) failed during the redundant second WRN model construction used only for architecture-count comparison, with `CUDA_ERROR_ILLEGAL_ADDRESS` surfacing from the pinned JAX/Flax initialization path. The smoke worker now uses the single actual WRN-28-10 construction and records its parameter count/configuration instead of initializing a duplicate model; this does not change the SAM/SGD implementation or settings. Runtime confirmation remains pending.

### Before starting

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

### First run: setup and preflight only

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

### Full runs only after preflight passes and costs fit

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

### Monitor, disconnect and retrieve

```bash
# On csci567-sam:
tmux attach -t sam-wrn
# Ctrl-b, then d detaches. Closing the laptop does not end tmux.
tail -f /workspace/sam_paper_replication/launcher_logs/YOUR_LOG.log
```

A finished tmux session disappears; its log and `launcher_*.json` receipt remain. For a graceful stop, attach and press Ctrl-c; allow worker cleanup and shelving to finish. Do not kill the supervisor or reset its ledger reservation. Jobs survive SSH loss, not a VM shelf/reboot/deletion. Results remain on the persistent volume after shelving. Reconnect later to download needed results, then run `~/bin/js2 stop csci567-sam` from WSL if still active. The SU allocation and the experiment's worker-time ledger are different budgets; keep both within limits.


Detailed operating and chat-handoff guide: `docs/jetstream_runbook.md` (added November 17, 2026).
