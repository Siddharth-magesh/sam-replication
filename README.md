## Current run instructions

[A100 one-epoch SGD/SAM/IGR timing (Python/tmux)](experiments/cifar100_resnet18_a100_one_epoch_timing/instructions.md) | [Completed CIFAR-10 results](docs/results/js2_50epoch_results_analysis.md) | [Documentation index](docs/README.md)

# SAM replication and input-gradient regularization experiments

This CSCI 567 project builds on the original `google-research/sam` JAX/Flax implementation. It includes a WideResNet-28-10 / CIFAR-10 reproduction workflow and a separate ResNet-18 / CIFAR-100 comparison of SGD, SAM and Input Gradient Regularization (IGR).

## New CIFAR-100 notebooks

The experiment notebooks belong in `repro_notebooks/`. The objective is to compare clean accuracy, input sensitivity, weight-space sharpness and adversarial accuracy. They are preliminary experiments, not reproductions of the paper's reported WideResNet results.

| Notebook | Training subset | Final training | Lambda selection | Status |
|---|---|---|---|---|
| [06: small pilot](https://github.com/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/06_cifar100_resnet18_sam_igr_pilot.ipynb) | 1,000 images, 10 per class | 2 epochs, seed 42 | 1 epoch each for 0, 0.01, 0.1 | A user-run T4 pilot completed all three methods and metrics. Functional check only. |
| [07: intermediate comparison](https://github.com/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/07_cifar100_resnet18_sam_igr_t4_intermediate.ipynb) | 5,000 images, 50 per class | 10 epochs, seed 42 | 3 epochs each for 0, 0.01, 0.1 | User-run T4 intermediate completed all ten epochs and metrics; optimizer/BN serialization check passed. Interrupted-training resume remains unverified. |

For **A100 runtime planning before the full experiment**, use [08: one epoch per method](https://github.com/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/08_cifar100_resnet18_a100_one_epoch_timing.ipynb). It uses 45,000 training / 5,000 validation images, batch 128, and exactly one measured epoch each of SGD, SAM and IGR after discarded warm-up. A100 with at least 38,000 MiB is required. It exports actual epoch/validation/checkpoint times and a configurable projection for 100 epochs and three seeds; it never launches that projected study. New A100 timing execution is pending. In Colab upload the local notebook; on the course Linux Jupyter instance set `WORK_ROOT` to a persistent mounted path. `RUN_TIMING=True` starts only the bounded timing run.

For the teammate handoff, use [A100 timing instructions and audited T4 observations](https://github.com/Siddharth-magesh/sam-replication/blob/repro/experiments/cifar100_resnet18_a100_one_epoch_timing/background.md). It includes the exact course-machine configuration, expected ZIP files and budget assumptions.

**Notebook 07 remains the intermediate T4 experiment.** It adds ten warm-up updates before twenty timed updates, a timing-stability check, validation input-sensitivity curves and optimizer/BatchNorm checkpoints after every completed epoch. The earlier pilot's short benchmark included a very slow first measured update and overestimated sustained training cost; do not treat notebook 06's timing projection as a reliable full-run estimate.

### Run notebooks 06/07 on free Colab T4

1. Open [notebook 06 in Colab](https://colab.research.google.com/github/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/06_cifar100_resnet18_sam_igr_pilot.ipynb) or [notebook 07 in Colab](https://colab.research.google.com/github/Siddharth-magesh/sam-replication/blob/repro/repro_notebooks/07_cifar100_resnet18_sam_igr_t4_intermediate.ipynb). You can also upload the local `.ipynb` file.
2. Select a **T4 GPU** runtime. Set `USE_DRIVE=True` before setup if results/checkpoints should survive deletion of the Colab runtime.
3. Run cells in order. Setup creates or reuses an isolated Python 3.8 / JAX 0.2.6 / CUDA jaxlib 0.1.57 / Flax 0.2.2 environment. Colab's own Python is unchanged; no local clone or runtime restart is needed.
4. Review the preflight output and timing estimate. `RUN_EXPERIMENT=True` is enabled by default, so running all cells starts training if the checks and one-hour estimate gate pass. Set it to `False` for preflight/timing only.
5. Download the result ZIP before ending the session. ZIP exports exclude large model and optimizer checkpoints; those remain in the printed run folder or Drive.

For **notebook 07** recovery, set `RESUME_RUN_ID` to the printed result-folder name and rerun with the same configuration, source and GPU, keeping the same `USE_DRIVE` setting. Completed epochs are restored; a partial epoch is replayed. Notebook 06 does not implement training resume. The one-hour cap applies per worker invocation; it does not disconnect Colab or cap account billing.

### Shared protocol and limits

- CIFAR ResNet-18 uses a 3×3 stride-1 stem, no max-pool and 100 outputs. All methods use the same seeded initialization, per-epoch data order and augmentation.
- Notebooks 06/07 use batch 32; notebook 08 measures batch 128. All use FP32, ordinary SGD momentum 0.9 (Nesterov off), LR 0.1 with cosine decay, L2 weight decay 0.0005 and global gradient clipping at 5. SAM uses global L2 radius 0.05.
- Validation contains 5,000 images, disjoint from training. In notebooks 06/07, lambda is selected by final tuning-epoch validation accuracy, with ties favoring the smallest value. All final methods then train from scratch. Test data never select lambda.
- IGR penalizes the mean per-image squared input gradient of cross-entropy with respect to raw [0,1] pixels, using exact nested autodiff. Its penalty branch freezes BatchNorm running statistics to avoid batch coupling; clean training cross-entropy updates BatchNorm normally. This convention is explicit and differs from penalizing a batch-coupled training loss.
- In notebooks 06/07, clean final test accuracy uses all 10,000 test examples. Notebook 08 uses validation only and does not load official test images. Input sensitivity, PGD-10 and the weight-sharpness proxy share a fixed balanced 100-image test subset. PGD uses L-infinity epsilon 8/255 and step 2/255. Sharpness uses global L2 radius 0.05, three ascent steps, one random start and fixed BatchNorm statistics.
- Sharpness is an approximate, parameterization-sensitive loss-increase proxy. The small PGD probe is not a robustness certificate. One seed and short training do not establish statistical significance or causal mechanisms. The proposed full-data, 100-epoch, three-seed experiment is not launched by these notebooks.

Project decisions and technical details are maintained in [project tracking](https://github.com/Siddharth-magesh/sam-replication/blob/repro/docs/project_tracking.md) and [notes](https://github.com/Siddharth-magesh/sam-replication/blob/repro/docs/notes.md). The original upstream README follows below.

---

# SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization

by Pierre Foret, Ariel Kleiner, Hossein Mobahi and Behnam Neyshabur.


## SAM in a few words

**Abstract**: In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by the connection between geometry of the loss landscape and generalization---including a generalization bound that we prove here---we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently. We present empirical results showing that SAM improves model generalization across a variety of benchmark datasets (e.g., CIFAR-{10, 100}, ImageNet, finetuning tasks) and models, yielding novel state-of-the-art performance for several. Additionally, we find that SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specifically target learning with noisy labels.


|      ![fig](figures/summary_plot.png)    |  ![fig](figures/no_sam.png)   |  ![fig](figures/sam_wide.png)     |
|:--------------:|:----------:|:----------------------:|
| Error rate reduction obtained by switching to SAM. Each point is a different dataset / model / data augmentation | A sharp minimum to which a ResNet trained with SGD converged | A wide minimum to which the same ResNet trained with SAM converged. |



## About this repo

This code allows the user to replicate most of the experiments of the paper, including:

 * Training from scratch Wideresnets and Pyramidnets (with shake shake / shake drop) on CIFAR10/CIFAR100/SVHN/Fashion MNIST, with or without SAM, with or without cutout and AutoAugment.
 * Training Resnets and Efficientnet on Imagenet, with or without SAM or RandAugment.
 * Finetuning Efficientnet from checkpoints trained on Imagenet/JFT on imagenet.


## How to train from scratch

Once the repo is cloned, experiments can be launched using sam.sam_jax.train.py:

```
python3 -m sam.sam_jax.train --dataset cifar10 --model_name WideResnet28x10 \
--output_dir /tmp/my_experiment --image_level_augmentations autoaugment \
--num_epochs 1800 --sam_rho 0.05
```

Note that our code uses all available GPUs/TPUs for training.

To see a detailed list of all available flags, run python3 -m sam.sam_jax.train --help.

#### Output

Training curves can be loaded using TensorBoard. TensorBoard events will be
saved in the output_dir, and their path will contain the learning_rate,
the weight_decay, rho and the random_seed.

## Finetuning EfficientNet:

We provide a FLAX checkpoint compatible with our implementation for all checkpoints
available on [official tensorflow efficientnet implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet):

|               |   B0    |  B1   |  B2    |  B3   |  B4   |  B5    | B6 | B7 | B8 |  L2 |
|----------     |--------  | ------| ------|------ |------ |------ | --- | --- | --- | --- |
| Baseline preprocessing |  76.7% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b0/checkpoint.tar.gz))   | 78.7% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b1/checkpoint.tar.gz))  | 79.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b2/checkpoint.tar.gz)) | 81.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b3/checkpoint.tar.gz)) | 82.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b4/checkpoint.tar.gz)) | 83.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckpts/efficientnet-b5/checkpoint.tar.gz)) | | || | |
| AutoAugment (AA) |  77.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b0/checkpoint.tar.gz))   | 79.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b1/checkpoint.tar.gz))  | 80.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b2/checkpoint.tar.gz)) | 81.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b3/checkpoint.tar.gz)) | 82.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b4/checkpoint.tar.gz)) | 83.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b5/checkpoint.tar.gz)) |  84.0% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b6/checkpoint.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/ckptsaug/efficientnet-b7/checkpoint.tar.gz))  || |
| RandAugment (RA) |  |  |  |  |  | 83.7%  ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/randaug/efficientnet-b5-randaug/checkpoint.tar.gz)) |  | 84.7%  ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/randaug/efficientnet-b7-randaug/checkpoint.tar.gz)) |  | |
| AdvProp + AA | 77.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b0/checkpoint.tar.gz)) | 79.6% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b1/checkpoint.tar.gz))  | 80.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b2/checkpoint.tar.gz)) | 81.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b3/checkpoint.tar.gz)) | 83.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b4/checkpoint.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b5/checkpoint.tar.gz)) | 84.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b6/checkpoint.tar.gz)) | 85.2% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b7/checkpoint.tar.gz)) | 85.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/advprop/efficientnet-b8/checkpoint.tar.gz))|| |
| NoisyStudent + RA | 78.8% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b0/checkpoint.tar.gz)) | 81.5% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b1/checkpoint.tar.gz)) | 82.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b2/checkpoint.tar.gz)) | 84.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b3/checkpoint.tar.gz)) | 85.3% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b4/checkpoint.tar.gz)) | 86.1% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b5/checkpoint.tar.gz)) | 86.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b6/checkpoint.tar.gz)) | 86.9% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-b7/checkpoint.tar.gz)) | - |88.4% ([ckpt](https://storage.googleapis.com/gresearch/sam/efficientnet_checkpoints/noisystudent/efficientnet-l2/checkpoint.tar.gz)) |

<sup>* We report in this table the scores as found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). If you use these checkpoints with another codebase, you might obtain some slightly different scores based on the type of accelerator you use and your data processing pipeline (resizing algorithm in particular).</sup>

<sup>* Advprop requires some slight modification of the input pipeline. See [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) for more details. </sup>

<sup>* Please refer to the README of the [official tensorflow efficientnet implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) to see which paper should be cited for which checkpoint.</sup>

Once the checkpoint is downloaded and uncompressed, it can be finetuned on any of the datasets:

```
python3 -m sam.sam_jax.train --output_dir /tmp/my_finetuning_experiment \
--dataset imagenet --from_pretrained_checkpoint true \
--efficientnet_checkpoint_path /tmp/path_to_efficientnet_checkpoint_to_finetune \
--learning_rate 0.1 --model_name efficientnet-l2-475 --batch_size 512 \
--num_epochs 10 --gradient_clipping 1.0 --label_smoothing 0.1 --sam_rho 0.1
```


## Bibtex

```
@ARTICLE{2020arXiv201001412F,
       author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
        title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
         year = 2020,
          eid = {arXiv:2010.01412},
       eprint = {2010.01412},
}
```

**This is not an official Google product.**
