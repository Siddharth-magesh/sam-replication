# Input-space stability versus weight-space sharpness: completed experiment

Prepared for the November 20 report review. Run `20261120_043937_4863bf`; protocol `cifar100_resnet18_sam_igr_full_v3_centered_bn`; launcher commit `f494329` on `repro`. All three tuning trials and all nine final training/evaluation runs completed.

**Main finding:** SAM improved clean test accuracy and reduced measured input-gradient sensitivity relative to SGD. However, validation selected **IGR lambda = 0**, so the final IGR arm is a zero-penalty control. This study does **not** establish that active input-gradient regularization reproduces SAM's benefit, nor that input stability causes SAM's gain.

## What was actually run

| Setting | Actual configuration |
|---|---|
| Model and data | CIFAR ResNet-18; CIFAR-100 |
| Split | Balanced 45,000 training / 5,000 validation images; split seed 2026 |
| Official test | 10,000 images; evaluated after tuning and final training |
| Final methods | SGD, SAM, IGR with selected lambda 0 |
| Final seeds | 0, 1, 2; paired initializations and epoch orders |
| Duration | 100 epochs per run; final-epoch weights |
| Batch | 128; 351 updates/epoch, dropping 72 shuffled training images |
| Optimizer | Momentum SGD, momentum 0.9, Nesterov off |
| Learning rate | 0.1 with cosine decay over 100 epochs |
| Weight decay / clipping | 0.0005 / global gradient norm 5 |
| SAM radius | 0.05, absolute global L2 |
| IGR definition | Mean squared per-image cross-entropy input-gradient norm in raw [0,1] pixel coordinates; nested differentiation to model parameters |
| IGR BN convention | Frozen running statistics in the penalty; batch statistics in clean training loss |
| Tuning | Lambdas 0, 0.01, 0.1; development seed 42; 100 epochs each; highest final validation accuracy, ties smallest lambda |
| Hardware/runtime | A100-SXM4-40GB; pinned Python 3.8 / JAX 0.2.6 / Flax 0.2.2 |

The optional SAM+IGR arm was **not run**. The alternative ResNet-20/CIFAR-10 prediction-consistency experiment in the proposal was **not run**. Do not describe this experiment as either of those designs. Full configuration and source hashes are in [run_metadata.json](run_metadata.json).

## Tuning outcome: essential limitation

| IGR lambda | Final validation accuracy (%) |
|---|---:|
| **0 (selected)** | **76.92** |
| 0.01 | 58.70 |
| 0.1 | 57.20 |

All three v3 tuning trials completed; none were rejected. Selection used validation data only. Nonzero penalties performed poorly under this grid and training convention; this is reportable negative tuning evidence, not a test of every possible IGR implementation or strength. Input-sensitivity and official-test metrics for these nonzero development trials were not evaluated, so we cannot claim that they reduced sensitivity without improving generalization.

Final IGR at lambda zero bypasses the input-gradient penalty and has the SGD objective. It remains labeled `IGR` in the original CSVs for provenance, but all plots/report text should label it **IGR (lambda=0 control)**. Small differences from SGD do not imply a regularization benefit: GPU deterministic controls were disabled, and the trajectories were not bitwise identical despite paired initialization/data order. Attributing their exact differences to a particular kernel is (to verify).

## Final measured results

Values are mean ± sample SD over three seeds; accuracy units are percent.

| Metric | SGD | SAM | IGR (lambda=0 control) |
|---|---:|---:|---:|
| Clean test accuracy | 76.4267 ± 0.2984 | **77.0733 ± 0.0709** | 76.5233 ± 0.2937 |
| Test cross-entropy | 0.93828 ± 0.00415 | **0.87045 ± 0.00812** | 0.94123 ± 0.00685 |
| Mean input-gradient L2 | 17.9315 ± 0.5188 | **15.1084 ± 0.2205** | 18.2580 ± 0.4629 |
| Mean squared input-gradient L2 | 1062.00 ± 86.40 | **759.68 ± 28.19** | 1111.35 ± 50.03 |
| Sharpness proxy, radius 0.025 | 0.11489 ± 0.01019 | 0.09504 ± 0.00409 | 0.11367 ± 0.01046 |
| Sharpness proxy, radius 0.05 | 0.33757 ± 0.08422 | 0.25939 ± 0.01903 | 0.29027 ± 0.03195 |
| PGD-20 accuracy (%) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

| Seed | SGD accuracy | SAM accuracy | IGR (lambda=0) accuracy | SAM minus SGD (points) |
|---|---:|---:|---:|---:|
| 0 | 76.77 | 77.15 | 76.26 | +0.38 |
| 1 | 76.23 | 77.06 | 76.84 | +0.83 |
| 2 | 76.28 | 77.01 | 76.47 | +0.73 |

SAM's mean accuracy improvement was **0.6467 percentage points**. Its mean input-gradient L2 was approximately **15.74% lower** than SGD's. Lower mean sharpness values are descriptive: the paired sharpness intervals include zero.

| Paired contrast | Mean difference | Marginal t-based 95% CI |
|---|---:|---:|
| SAM − SGD, accuracy points | +0.6467 | [0.0597, 1.2336] |
| SAM − SGD, input-gradient L2 | −2.8231 | [−3.8302, −1.8159] |
| SAM − SGD, sharpness radius 0.05 | −0.07819 | [−0.27148, 0.11511] |
| IGR(0) − SGD, accuracy points | +0.0967 | [−1.3089, 1.5022] |
| IGR(0) − SAM, accuracy points | −0.5500 | [−1.3825, 0.2825] |

Intervals use three paired seed differences and Student t with two degrees of freedom. They have low precision, assume independent seed differences and are not adjusted for multiple comparisons. No equivalence margin was specified; overlapping results do not establish equivalence.

## Measurement and implementation disclosures

Clean accuracy uses the full official test set. Input sensitivity, PGD and sharpness use the same balanced 1,000-test-image probe (10 per class, seed 31415). Input gradients use frozen BN and the gradient of the sum of per-image losses, avoiding a 1/batch scaling error. PGD uses L-infinity epsilon 8/255, step size 2/255, 20 steps and three random restarts. All measured PGD accuracies were zero, so this attack does not demonstrate a robustness advantage for SAM; this is not a robustness certificate or an exhaustive attack evaluation.

Sharpness is maximum observed cross-entropy increase under absolute-L2 parameter perturbations, at radii 0.025 and 0.05, using ten ascent updates and three starts with BN frozen. It is a parameterization-sensitive optimization lower-bound proxy, not the exact worst-case sharpness or a causal measure. Parameter norms differ between methods; do not interpret absolute radii as normalized perturbations.

Earlier attempts failed numerically and showed poor inference accuracy. A training-only BN audit raised validation from 1.64% to 67.36% with unchanged weights. V3 therefore used centered-moment BN for all arms and recalibrated an evaluation-only BN copy using 100 fixed training batches before each validation. The original optimizer and training BN state were preserved separately. Final evaluation used the saved calibrated state. Raw and calibrated validation values are both retained in [epoch_metrics.csv](epoch_metrics.csv).

These decisions were made after inspecting validation diagnostics and must be disclosed as protocol revisions. Final official-test data were not used for lambda selection or calibration. V3 also bypassed the IGR penalty at lambda zero and used scaled gradient norms. Original package pins were retained; this is an adapted hypothesis experiment, not an unchanged reproduction of the paper's training code. Earlier v1/v2 failures remain part of the research record; the precise cause of their NaNs was not conclusively isolated. V3 completed successfully, which does not prove all future configurations stable.

## Interpretation against the proposed hypotheses

- **Hypothesis 1 / Experiment 2:** SAM's higher accuracy and lower input sensitivity are consistent with an association. The intended decisive comparison with an active IGR method was not achieved because lambda zero won selection. The alternative mechanism remains **inconclusive**, neither established nor disproved.
- **Prediction-consistency hypothesis (Hypothesis 3):** not directly tested. A prediction-consistency loss under perturbed inputs is not the same objective as this input-gradient penalty.
- **Matched-training-accuracy robustness hypothesis (Hypothesis 4):** not tested as specified. Runs were matched by configuration/epochs, not training accuracy; training accuracy was not logged. The chosen PGD evaluation yielded zero for every arm.
- **Weight-space-flatness claim:** SAM had lower mean proxy sharpness, but paired intervals were wide and included zero. Cross-method associations cannot establish mediation or causality within SAM.

Suggested draft paragraph:

> On CIFAR-100 with ResNet-18, SAM achieved 77.07 ± 0.07% clean test accuracy versus 76.43 ± 0.30% for SGD across three seeds, with a mean paired gain of 0.65 percentage points. SAM also had lower mean input-gradient sensitivity. However, validation selected zero input-gradient regularization, so the final IGR arm did not constitute an active intervention on input stability. Our results therefore support a SAM–input-stability association but leave the proposed alternative mechanism unresolved. Nonzero IGR settings performed poorly during validation tuning, and the evaluation protocol required training-only BatchNorm recalibration after earlier numerical failures.

## Post-hoc validation sensitivity diagnostic

On November 20, evaluated the saved epoch-100 tuning models (development seed 42) on the same balanced 1,000 validation images, 10 per class, probe seed 271828. Used each model's saved calibrated BN. No retraining, test-set access or lambda re-selection occurred; original checkpoint, model/BN, optimizer and study evidence integrity checks passed.

| IGR lambda | Validation accuracy (%) | Mean input-gradient L2 | Mean squared L2 | L2 reduction vs zero |
| --- | --- | --- | --- | --- |
| 0 | 76.70 | 19.2810 | 1173.0662 | 0% |
| 0.01 | 57.10 | 12.3270 | 302.3573 | 36.07% |
| 0.1 | 57.70 | 7.9026 | 108.1007 | 59.01% |

Active IGR reduced the measured input sensitivity but reduced validation accuracy by 19.6/19.0 percentage points in these tuning models. It did not reproduce the desired accuracy benefit at these settings. This weakens the proposed operational prediction for this configuration; it neither identifies SAM's causal mechanism nor rules out other regularization strengths. This is a post-hoc, single-development-seed result, not a three-seed test. These subset accuracies are distinct from the original 5,000-image tuning scores and official-test results. No new weight-sharpness measurement was made.

Data: [validation_comparison.csv](validation_probe/validation_comparison.csv), [comparison.json](validation_probe/comparison.json), [status.json](validation_probe/status.json), [provenance.json](validation_probe/provenance.json). Attempt `20261120_230049_d80b2d` completed in tmux with exit code 0: 76.71 seconds (0.0213094 hours), cumulative study ledger now 7.430089 hours. Earlier accounting below remains the pre-probe snapshot. Idle GPU allocation and transfer time are excluded.

Shareable local ZIP: `experimental_results/js2_cifar100_full/20261120_230049_d80b2d_validation_probe.zip`. SHA-256 verified against remote: `acfa5426a9b129aff95c971e6da84175affe446ae63d6698bd31656f050d60f7`. Includes per-model logs/metrics, subset indices and source snapshots. A separate launcher log is saved alongside it. The instance remains active following this diagnostic.

## Runtime, files and provenance

The successful v3 invocation took **5.3746 hours**, covering tuning, training, evaluation and orchestration. The cumulative ledger was **7.4088 hours**, including **2.0342 hours** of earlier attempts/diagnostics; do not add these totals together. The approximately 16.22-hour planning estimate was conservative, not measured runtime. Idle allocation, archive transfers and unrelated Colab setup are excluded. Do not add this cumulative ledger blindly to the separate 50-epoch ledger without reconciling their scope.

- [per_seed_metrics.csv](per_seed_metrics.csv), [mean_std_metrics.csv](mean_std_metrics.csv), [paired_differences.csv](paired_differences.csv): original final statistical tables.
- [epoch_metrics.csv](epoch_metrics.csv): all 1,200 epoch records, including raw/calibrated validation and timing.
- [lambda_selection.json](lambda_selection.json): all development scores and selection rule.
- [runtime_by_job.csv](runtime_by_job.csv), [compute_summary.json](compute_summary.json): measured runtime and accounting.
- [run_metadata.json](run_metadata.json), [provenance.json](provenance.json), [summary.json](summary.json): configuration, source/archive hashes and completion evidence.
- [Run instructions](../../experiments/cifar100_resnet18_sam_igr_full/instructions.md): implementation and execution details.

Local full backup: `experimental_results/js2_cifar100_full/20261120_043937_4863bf_full.tar` (12 checkpoints). Local shareable evidence ZIP: `experimental_results/js2_cifar100_full/20261120_043937_4863bf_evidence.zip`. Both hashes matched the remote copies; the nine final evaluations and twelve checkpoints were verified. Raw archives are excluded from Git; this compact package is the teammate reference. The instance was shelved after download.
