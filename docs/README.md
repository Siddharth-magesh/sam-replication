# Project documentation

- [Final 50-epoch result package](../results/js2_50epoch_summary/README.md): CIFAR-10 / WRN-28-10, one seed, completed reduced replication.
- [Completed hypothesis experiment](../results/cifar100_resnet18_sam_igr_100epoch/README.md): CIFAR-100 / ResNet-18, three seeds; lambda-zero selection makes the active-IGR mechanism comparison inconclusive.

- [Project tracking](project_tracking.md): experiment status and history.
- [Technical notes](notes.md): implementation decisions and evidence.
- [Coding guidelines](project_coding_guidelines.md): repository conventions.
- [Completed original JAX results](results/js2_50epoch_results_analysis.md): 50-epoch CIFAR-10 / WRN comparison.

Run instructions live with their experiment:

- [Next: Full A100 SGD/SAM/IGR study](../experiments/cifar100_resnet18_sam_igr_full/instructions.md).
- [Previous: A100 SGD/SAM/IGR one-epoch timing](../experiments/cifar100_resnet18_a100_one_epoch_timing/instructions.md).
- [Previous: CIFAR-10 WRN SAM/SGD, 50 epochs](../experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/instructions.md).
- [Historical Jetstream operations/debugging](../experiments/cifar10_wrn28x10_sam_sgd_50epochs_seed0/jetstream_runbook.md).

Historical diagnostics are retained under [archive](archive/colab_a100_model_init_diagnostic.md); they are not the current run procedure. Existing executed notebooks remain preserved.
