# Independent PyTorch SAM / SGD validation track

**No measured speedup or runtime estimate exists yet. Do not interrupt the running JAX job.** This branch is a separately labeled implementation. FP32 CPU checks passed; full WRN initial-weight/output parity, JAX update parity, CUDA/compile/BF16 behavior and A100 performance are **(to verify)**. Training is gated on reviewed parity evidence; candidate optimizations are not automatically approved.

Model parameter count derives from layer shapes: **36,220,634**, confirmed on the full PyTorch model. Small controlled CPU WRN checks found zero discrepancy from the independent autograd oracle for SGD/SAM gradients, zero repeat-run variation and zero BN-state difference. Vectorized relative gradient errors were 3.13e-7 (SGD) and 1.49e-6 (SAM). Fixed CPU arithmetic tolerances were declared as atol=2e-5, rtol=2e-4 before these measurements; they are **not calibrated JAX/GPU tolerances**. BN decay/biased variance, augmentation reflection/order, explicit L2 placement, clipping, Nesterov updates, parameter restoration and checkpoint continuation were checked. A CPU-only PyTorch 2.9.0 venv outside the repo was used; no GPU code ran.

`models/` implements the source-specific WRN; `optimizers/` implements per-lane SAM and shared updates; `data/` provides paired augmentation/permutations; `training/` checkpoints/evaluates; `validation/` exports and compares legacy tensors; `benchmarks/` contains bounded execution and candidates; `scripts/` holds wrappers/setup. See [source decisions](references.md).

## Environment and CPU checks

Run from this branch's checkout. Paths below are **new**, separate from all JAX/general environments. CPU validation needs no dataset or GPU.

```bash
git fetch origin
git switch fast-pytorch-sam
git pull --ff-only origin fast-pytorch-sam
python3 -m fast_torch.scripts.setup_environment --cpu --prefix /workspace/fast_torch_cpu_env --execute
/workspace/fast_torch_cpu_env/bin/python -B -m fast_torch.validation.parity_tests --output /workspace/fast_torch_cpu_checks.json
```

Only **after JAX finishes and the GPU is idle**, create the GPU environment. The script inspects driver/GPU before choosing the official cu126 wheel, requires an A100, and refuses existing prefixes. It installs no packages into existing environments.

```bash
python3 -m fast_torch.scripts.setup_environment --prefix /workspace/fast_torch_gpu_env --execute
source /workspace/fast_torch_gpu_env/bin/activate
```

## Legacy export and cross-framework validation

These commands are prepared but **not executed (requires legacy env)**. Do not run during JAX training. Export uses the original pinned runtime source and original update, not the independent implementation. `--updates` additionally exports both original eight-lane updates and repeat-run noise. Artifacts may be large and must remain outside Git.

```bash
mkdir -p /workspace/fast_torch_artifacts
PYTHONPATH=/workspace/sam_paper_replication/work \
LD_LIBRARY_PATH=/workspace/sam_paper_replication/envs/sam/lib \
PATH=/workspace/sam_paper_replication/envs/sam/bin:$PATH \
TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_cuda_data_dir=/workspace/sam_paper_replication/cuda --xla_gpu_deterministic_reductions=true' \
/workspace/sam_paper_replication/envs/sam/bin/python fast_torch/validation/export_jax_weights.py \
  --seed 0 --updates --output /workspace/fast_torch_artifacts/seed0.npz

python -B -m fast_torch.validation.compare_initialization \
  /workspace/fast_torch_artifacts/seed0.npz --updates \
  --output /workspace/fast_torch_artifacts/jax_comparison.json
```

Comparison defaults to CPU, records parameter/layout equality, output differences, updated parameters/deltas and BN differences. Review original repeatability before deriving tolerance bounds; do not loosen bounds to obtain a pass. To certify against reviewed bounds, repeat comparison with `--calibration /workspace/fast_torch_artifacts/calibration.json`. That JSON contains `max_abs`, `relative_l2` and a nonempty `provenance` explaining measured noise and the predeclared numerical allowance. Without calibration the result remains `measured_not_certified`.

## Bounded benchmark, then training

Use the existing verified **binary** CIFAR-10 archive; set its actual path below. No downloads occur implicitly. Host-loaded and GPU-resident variants use the same CPU-generated plans; the host iterator currently has no multiprocessing/prefetch workers. Exact TensorFlow RNG matching is not claimed.

```bash
export CIFAR_ARCHIVE=/workspace/sam_paper_replication/tensorflow_datasets/downloads/manual/cifar-10-binary.tar.gz
python -B -m fast_torch.benchmarks.benchmark --execute --minutes 45 \
  --init /workspace/fast_torch_artifacts/seed0.npz --archive "$CIFAR_ARCHIVE" \
  --output /workspace/fast_torch_artifacts/benchmark_reference
```

To investigate individual candidates, use a fresh output folder and `--cases vectorized compile channels_last foreach fused gpu_data cudnn_benchmark tf32 bf16`. The 45-minute cap applies to the **whole invocation**, not each case; failures stop the invocation and unmeasured candidates remain unverified. No benchmark path starts full training. Live utilization/memory/power CSV, first-call/compile-inclusive and warmup timing, compiler instrumentation, synchronized ms/step, images/s, estimated epoch time and memory peaks are recorded. Candidate results are not parity certificates. No ghost-BN SGD shortcut or CUDA graphs are enabled.

Before full training, review the JAX comparison and same-input candidate parity plus benchmark measurements. Place a reviewed `approval.json` outside Git containing `approved: true`, `jax_parity_verified: true`, `case`, `source_sha256`, and `init_sha256` matching the benchmark identity. Non-reference cases additionally require `measured_faster_and_parity_passed: true`. Do not manufacture this evidence: these fields are deliberately unset because required GPU/JAX validation has not run. FP32 reference is the starting point.

```bash
python -B -m fast_torch.benchmarks.benchmark --mode train --execute --minutes 60 \
  --cases reference --methods SAM SGD --config fast_torch/configs/cifar10_50.json \
  --init /workspace/fast_torch_artifacts/seed0.npz --archive "$CIFAR_ARCHIVE" \
  --approval /workspace/fast_torch_artifacts/approval.json \
  --output /workspace/fast_torch_artifacts/comparison50
```

The 60-minute example is a cap, not a completion estimate; choose an approved total allocation after benchmarking. Resume with the same command plus `--resume`. Both methods start from the same imported seed-0 state; final epoch error is the reported endpoint. The 200-epoch config is available but unused. 50-epoch results are not directly comparable with the paper's 2.7%/3.5% results. Checkpointed momentum, parameters, eight BN states, epoch and step plus epoch-derived augmentation enable epoch-boundary recovery. Partial epoch time remains charged in the receipt.

Every bounded invocation refuses a busy GPU and saves an incremental `gpu_hours_*.json` receipt, including failed/partial work. Reconcile these receipts with the project's remaining 40-hour ledger **once**, without reimporting previous usage. The fast track never edits/resets the running JAX ledger. Allocation before/after execution is additional. No automatic shelving or remote lifecycle commands are issued.
