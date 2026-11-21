# Exploratory lower-lambda IGR results

Completed study `20261121_064811_a6db67`: three 100-epoch development trials and three fresh final IGR seeds, with six reused SGD/SAM evaluations. All jobs completed; no rejected candidates. Runtime: **4.2993 hours (4 h 18 min)** of new work, excluding idle allocation and transfer. Prior study usage is additional.

Validation selected lambda **0.00001**: 76.94% versus 74.08% for 0.0001 and 69.52% for 0.001. The previous zero-lambda reference was 76.92%; the 0.02-point development difference is not evidence of a reliable gain. Selection was among positive candidates by design, using full validation accuracy only.

| Method | Test accuracy, mean +/- sample SD (%) | Mean input-gradient L2 | Mean sharpness proxy, rho=0.05 |
|---|---:|---:|---:|
| SGD (reused) | 76.427 +/- 0.298 | 17.932 | 0.338 |
| SAM (reused) | 77.073 +/- 0.071 | 15.108 | 0.259 |
| IGR, lambda=0.00001 (new) | 76.650 +/- 0.061 | 14.972 | 0.595 |

IGR reduced mean input sensitivity by about 16.5% versus SGD and avoided the large accuracy collapse of the stronger penalties. Its accuracy gain over SGD was only 0.223 points, paired marginal t95 CI [-0.367, 0.813], so an improvement is not established. IGR was 0.423 points below SAM, CI [-0.499, -0.347]. Input sensitivity was numerically similar to SAM (difference CI includes zero; no equivalence claim), while its mean sharpness proxy was about 2.29 times SAM's (paired difference CI [0.021, 0.650]). All PGD accuracies were zero.

This demonstrates lower measured input sensitivity with higher measured sharpness, but does not establish that IGR reproduces SAM's generalization gain or identify SAM's causal mechanism. Three seeds, unadjusted marginal intervals, reused references and design after prior test exposure make this exploratory. Sharpness is an absolute-L2 parameterization-sensitive optimization lower-bound proxy. Same v3 centered BN, training-only evaluation BN calibration, data split, model and numerical worker were used; prior results remain unchanged.

## Evidence

- `per_seed_metrics.csv`, `mean_std_metrics.csv`, `paired_differences.csv`: final metrics and descriptive comparisons.
- `lambda_selection.json`, `tuning_diagnostics.csv`: all tuning outcomes and separate validation sensitivity probes.
- `summary.json`, `status.json`, `manifest.json`, `reference_provenance.json`: completeness, frozen configuration and reused-reference provenance.
- `incremental_budget.json`: extension-only active runtime receipts; add once to prior project accounting.
- `archive_verification.json`: downloaded ZIP checksum and verification record.

Local evidence ZIP: `experimental_results/js2_igr_lower_lambda/20261121_064811_a6db67_evidence.zip`. ZIP integrity and remote SHA-256 matched; all six training result records report 100 completed epochs and all three final IGR evaluations completed. Tables were recomputed from per-seed metrics. ZIP contains logs, source snapshots and histories; large checkpoints remain on the persistent JS2 volume and are not included in the ZIP or Git.
