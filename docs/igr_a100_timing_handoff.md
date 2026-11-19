# ResNet-18 / CIFAR-100: T4 results and A100 timing handoff

## What is ready

The SGD/SAM/IGR pipeline was tested on Colab's free Tesla T4. Notebook 06 completed a small functional pilot, and notebook 07 completed the intermediate comparison below. Notebook 08 is the next step: measure one complete epoch of each method on the A100 before budgeting longer training.

**The new A100 notebook passed local syntax, control-flow and report checks; it has not yet been GPU-tested on A100.** Successful T4 training does not establish A100 memory use or runtime.

Repository: [Siddharth-magesh/sam-replication, repro branch](https://github.com/Siddharth-magesh/sam-replication/tree/repro).

Notebook to run: [08_cifar100_resnet18_a100_one_epoch_timing.ipynb](https://github.com/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/08_cifar100_resnet18_a100_one_epoch_timing.ipynb).

## Completed intermediate experiment

Audited user-supplied run: `20261119_052637_f9ddec` (raw run bundle is not included with this guide). ResNet-18 / CIFAR-100, balanced 5,000 training images and 5,000 validation images, 10 epochs, batch 32, FP32, seed 42. Ordinary SGD momentum 0.9, LR 0.1/cosine, L2 0.0005, clipping 5, SAM rho 0.05. IGR lambda 0.01 was selected using validation accuracy from three-epoch trials at 0, 0.01 and 0.1; test data did not select lambda.

| Final metric | SGD | SAM | IGR |
|---|---:|---:|---:|
| Clean test accuracy (%) | 22.33 | 20.87 | 23.29 |
| Mean input-gradient L2 | 3.2243 | 2.7632 | 2.2546 |
| Weight-sharpness proxy, radius 0.05 | 0.1469 | 0.1077 | 0.3532 |
| PGD-10 accuracy (%) | 1 | 3 | 3 |

Clean test accuracy uses all 10,000 test images. The other metrics use the same balanced 100-image test subset. Sharpness uses only three ascent updates and one start; PGD uses epsilon 8/255 and step 2/255. These small probes are preliminary.

Observed differences: IGR was 0.96 percentage points above SGD, with about 30% lower input-gradient L2 and about 3.28 times SAM's measured sharpness. SAM was 1.46 points below SGD, so this run did **not** reproduce a SAM generalization advantage. The pattern motivates further investigation but does not establish the mechanism of SAM or statistical significance. Longer training and independent seeds could preserve, weaken or reverse these differences.

The main experiment worker, including lambda trials and final evaluation, took 1,056.24 seconds (17.60 minutes). Data download accounted for roughly 31 minutes in the separate preflight phase. Preflight checks passed, including exact optimizer/BN serialization roundtrip, lambda-zero consistency and nonzero-lambda update effect. Interrupted-training resume has not been tested on GPU.

## What the A100 test runs

| Setting | Value |
|---|---|
| GPU | A100 with at least 38,000 MiB visible GPU memory |
| Dataset | CIFAR-100, 45,000 train / 5,000 validation |
| Architecture | Same CIFAR ResNet-18 as notebooks 06/07 |
| Batch / precision | 128 / FP32 |
| Actual measured training | **One epoch each of SGD, SAM and IGR**, seed 42 |
| Epoch size | 351 full updates, 44,928 images; 72 shuffled examples dropped |
| Warm-up | 10 warm-up updates + 20 short-probe updates, then discard all warm-up state |
| Schedule | Epoch one of a 100-epoch cosine schedule |
| SGD | Ordinary momentum 0.9; LR 0.1; L2 0.0005; clipping 5 |
| SAM / IGR | rho 0.05 / lambda 0.01 for timing only |
| Validation | All 5,000 validation images, separately timed |
| Checkpoint | Optimizer + BN write and exact restore check, separately timed |

**It does not run 100 epochs, tune lambda, run three seeds, or evaluate the official test set.** Those larger settings are used only in the arithmetic budget projection. Batch 128 differs from the T4 pilot's batch 32, so this is a new runtime/memory check.

The scientific update code is retained from notebook 07. IGR uses exact mixed derivatives, with frozen BN in the input-gradient penalty branch and normal BN updates in the clean CE branch. Each method uses a separate subprocess. Initial model/BN and data-order hashes must agree before the combined projection is produced.

## Run on the course machine

1. Use the A100 when it is idle so another job does not distort timing. Do not interrupt an existing team experiment to run this benchmark.
2. From GitHub's `repro` branch, download notebook 08 and upload/open it in the machine's Jupyter interface. The notebook itself clones the pinned model source; it does not need to be launched from inside the repository.
3. In its first configuration cell, set:

   ```python
   RUN_TIMING = True
   USE_DRIVE = False
   WORK_ROOT = '/workspace/sam_igr_a100_timing'
   MAX_SESSION_MINUTES = 60
   MEASURE_CHECKPOINT = True
   ```

   `/workspace` should be your mounted persistent volume. If your mount differs, use a new writable directory on that volume. This root keeps the timing environment and outputs separate from the existing WRN experiment.

4. Run all cells in order. Setup creates or verifies its own pinned Python 3.8 / JAX 0.2.6 / CUDA jaxlib 0.1.57 / Flax 0.2.2 environment. Jupyter's system Python can remain unchanged. Setup also verifies the A100 and prepares the CIFAR cache before timing training.
5. Wait for one epoch each of SGD, SAM and IGR, then run the report and export cells. No additional training is triggered by the report.
6. Download the ZIP from the path printed by the final cell using Jupyter's file browser. On this machine its filename ends with `_a100_one_epoch_timing.zip`.

For Colab A100 instead, use the default `WORK_ROOT` and optionally `USE_DRIVE=True`. Drive mounting is specific to Colab.

All setup/data/training subprocesses share a 60-minute deadline from the setup cell. If installation/download consumes the deadline, keep the failed output and rerun from setup: the same `WORK_ROOT` reuses the environment/data cache while creating a fresh timing folder. The deadline does not stop GPU billing or shut down the instance. Stop an idle instance using the normal team workflow after collecting the output.

An OOM or failed method preserves its logs and prevents a complete projection. Do not silently reduce batch size or remove IGR; send the failed bundle so we can review a new shared configuration. There is no training-resume option in this one-epoch benchmark.

## What to send back

Send the exported ZIP, including:

- `measured_epoch_times.csv`: measured training, validation, checkpoint, overhead and total worker time for each method.
- `budget_projection.json` and `planning_scenario.json`: projected study cost and the exact assumptions.
- `SGD_result.json`, `SAM_result.json`, `IGR_result.json`: hardware/environment, timing distributions and checks.
- The three method logs/statuses, plus setup/data logs if anything failed.

The ZIP includes step timings and source/config hashes. Large `.msgpack` checkpoints remain in the output folder and are excluded from the ZIP; they are not needed to review timing. Preserve the ZIP before deleting the runtime or volume.

## Read the budget correctly

Defaults project 100 epochs x 3 seeds for each of SGD, SAM and IGR, plus **three additional 100-epoch IGR search trials on one development seed**. Every projected epoch includes validation and a checkpoint write. Fixed process/warm-up overhead is conservatively included once per run. There is also an explicitly unmeasured 60-minute reserve for later test/PGD/sharpness work and a 20% allowance.

`PLANNING_BUDGET_HOURS = 35` is only a comparison with a declared planning budget, not a query of the remaining course allocation. Change the `PLANNED_*` scenario if the team chooses a different tuning horizon; it still does not change the one-epoch worker loop. No optional SAM+IGR arm or additional model family is included.

These are first-epoch extrapolations, not measured full-study runtimes. We will review the actual measurements and remaining allocation before choosing the final training matrix. The A100 timing run does not establish which scientific hypothesis is correct.
