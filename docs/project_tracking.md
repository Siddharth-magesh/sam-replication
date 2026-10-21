# SAM Replication and Critical Analysis: Project Tracker

**Last updated:** 2026-10-20 | **Branch:** `repro` | **Paper:** Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization* (ICLR 2021) | **Code:** fork of `google-research/sam`

This file is the single place where the team can see what has been done, what is being done, and what comes next. It tracks progress and decisions. Technical details (the exact environment recipe, code findings, raw numbers) live in `NOTES.md`. Whoever finishes a task updates the status table in Section 9 and adds one line to the changelog in Section 12, in the same commit as their work. Anything marked **(to verify)** is an inference that nobody has confirmed yet.

---

## 1. Project at a glance

The course project asks us to pick a machine learning paper, identify its one central claim, examine the experiments that support it, propose an alternative hypothesis that would produce the same results, design a small experiment on which the two explanations disagree, run it, and report the outcome in a 4-page report. We are not asked to beat the state of the art. We are graded on the quality of our reasoning and the honesty of our reporting, and all three outcomes (paper confirmed, alternative confirmed, inconclusive) are equally acceptable. The project is 20% of the course grade, done in a group of 3 to 4.

Our paper is SAM. Our compute is one A100 (40 GB) per team through the course allocation, with a guideline of about 100 GPU-hours for the whole semester. The grading weights are: understanding of the paper 20%, critical examination of the evidence 25%, alternative hypothesis 15%, experimental design 20%, execution and reporting 15%, writing 5%. The critical examination and the experimental design together are 45% of the grade, so the thinking matters more than the volume of compute we burn.

## 2. Timeline and where we stand

| Milestone | Date | What is required | Status |
|---|---|---|---|
| Topic and team | Fri Oct 9 | Team, paper, 2-3 sentences on the central claim | Done |
| Project check-in | Fri Oct 30 (discussion section) | Paper read closely, central claim in one sentence, key experiments described, draft alternative hypothesis, paper's code run or set up, one-page summary for the TAs | **In progress** |
| Report review | Fri Nov 20 (discussion section) | Full draft with the experiment designed and preliminary results | Not started |
| Final report | ~Thu Dec 17 | 4-page report plus appendix and a link to our code, on Gradescope | Not started |

The check-in is ten days away. The code side of it is largely covered (Section 4). The reading and hypothesis side is not, and it needs the most attention this week.

## 3. The paper and our working central claim (draft, to be agreed by the whole team)

SAM replaces the usual training objective with a min-max objective: instead of minimizing the training loss at the current weights, it minimizes the *worst-case* training loss within a small neighborhood of radius rho around them. In practice each step computes the gradient, takes a step of norm rho in the gradient direction to a perturbed point, computes the gradient there, and applies that gradient to the original weights. This costs two forward/backward passes per step.

Our draft of the central claim, separated into its two parts as the assignment asks: the **what** is that training with SAM gives lower test error than standard SGD training across a range of vision models and datasets; the **why** is that it does so because it finds flat minima, meaning low loss across a whole neighborhood, and flatness is what causes better generalization. A useful test for whether this is the right claim: if the "why" turned out to be false, would the paper's contribution collapse? The "what" would survive as an engineering result, but the paper's framing, its theorem, and its section on m-sharpness would lose their purpose. Each of us should read the paper independently and write down the central claim in our own words before the next team meeting. If our versions differ, that is itself informative, and we should resolve it before the check-in.

The load-bearing evidence is Table 1 (CIFAR-10/100, five seeds each, with SAM beating SGD in every row), Table 2 (ImageNet), Table 3 (fine-tuning), Table 4 (label noise), Figure 3 (Hessian spectra and m-sharpness), and the ablations in Appendix C.

## 4. What has been done

**Repository and reading.** The group fork (`sam-replication`, branch `repro`) exists. We have read `train.py` and `flax_training.py` line by line for flags, defaults, the SAM update, and the output format. We confirmed this is the authors' official release (it is linked from the paper's abstract and carries the authors' Apache 2.0 header), with the caveat that official does not guarantee it is byte-identical to the code that produced every number in the paper.

**Environment.** The code is a 2020 JAX/Flax stack that needs Python 3.8 and pinned libraries, so it does not install on a current Colab as-is. We built a working environment through four fixes, each recorded in `NOTES.md`: (1) an `absl-py` pin in `requirements.txt` that conflicts with `tensorflow-metadata`, relaxed from 0.11.0 to 0.10.0; (2) a CUDA build of `jaxlib` (0.1.57+cuda110) that must come from a separate wheel index and must be installed before the other packages, or pip silently installs a CPU-only version; (3) a missing `libdevice` file that XLA looks for in a fixed path, solved with a symlink; and (4) cuDNN 9 being installed by default while this `jaxlib` needs cuDNN 8, solved by pinning `cudnn=8.0.5`. The symptom of that last one was "DNN library is not found", which is easy to misdiagnose.

**Data.** CIFAR-10 downloads from `cs.toronto.edu` at very low speed (about 38 minutes for 162 MB). We downloaded it once and cached the processed TFDS folder on Drive, so nobody else has to wait.

**Smoke tests and timings.** On a Colab T4 (15 GB) we ran SAM (rho 0.05) for 2 epochs and plain SGD (`--sam_rho 0.0`) for 1 epoch on WideResNet-28-10 with the paper's CIFAR hyperparameters. Both trained correctly: SAM's test error fell from 46.0% to 35.1% over two epochs, and SGD reached 43.5% after one. SAM took roughly 457 to 480 seconds per epoch against 257 seconds for SGD (about 1.8x slower), plus 17 to 20 seconds for evaluation. GPU memory use for SAM was 13.6 of 15 GB. **These runs prove the pipeline works. They are not a replication and must not be compared with each other**, because the cosine learning-rate schedule stretches over the number of epochs, so the 1-epoch and 2-epoch runs followed different schedules.

**Shared notebook.** A clean Colab notebook with the tested setup, the data download, and the two smoke tests is ready for teammates to confirm that the code runs on their own accounts.

## 5. Findings from the code that affect the replication

These are documented in detail in `NOTES.md`; the ones that change how we interpret results are summarized here.

**Defaults differ from the paper.** The default batch size is 128 and the paper uses 256. The default weight decay is 0.001, while the paper's WideResNet-28-10 / 200-epoch row uses 0.0005 (0.001 belongs to the 1800-epoch row). Gradient clipping at global norm 5.0 is on by default for both SAM and SGD, and the paper never mentions it. We always pass batch size and weight decay explicitly.

**One GPU changes the experiment (m = 256 vs m = 32).** The paper trained on 8 GPUs with unsynchronized perturbations, so each GPU computed its own SAM perturbation over its 32 examples. The paper's own Figure 3 (middle) shows smaller m gives better results. Batch normalization in this code has no `axis_name`, so its statistics are also computed per device. On our single GPU, both the perturbation and the batch norm statistics use all 256 examples. So a single-GPU run differs from the paper's setup in two linked ways, and our SAM gain may come out smaller than 2.7 vs 3.5.

**Seeds probably do not give independent runs (to verify).** The weight initialization uses a hardcoded `PRNGKey(0)` in `load_model.py`, the dataset shuffle in `dataset_source.py` is unseeded, and `--run_seed` only seeds stochastic layers that WideResNet-28-10 does not use. If that holds, five "seeds" share the same starting weights. The paper says its five replicas had independent initialization and shuffling. We need to patch this before running multiple seeds, or our confidence intervals will be misleadingly narrow.

**Evaluation uses the test set.** The log says "Used test set instead of validation set", while the paper says rho was tuned on a 10% validation split. Reporting the final-epoch test error (not the best epoch) avoids selecting on the test set, and we should say plainly in the report that the released code does not hold out a validation split.

**SAM's logged training metrics are measured at the perturbed weights**, so they cannot be compared with SGD's training metrics. Only test metrics are comparable.

## 6. First observations on the paper's evidence

These are raw material for Section 2 of the report ("the evidence and its weak points"). They are observations from a first read, not conclusions, and each needs a second look before it goes into the report.

**Baseline treatment.** SGD baselines get up to twice the epochs, and the paper reports the better of the standard and doubled count, which is fair on compute. But the paper says learning rate and weight decay were grid-searched before SAM was applied, and rho was then tuned over six values on top. Appendix Table 9 shows that a fixed rho of 0.05 still beats SGD in most CIFAR rows, which partly answers the tuning concern, so the asymmetry is real but its effect may be small.

**Selection and consistency.** The CIFAR results are consistent across 15 rows, but the SVHN table (Appendix Table 5) has one row, Shake-Shake with AutoAugment, where SAM is slightly worse (1.07 vs 1.03). The main text describes SVHN results as "at or above" the state of the art, so the reader should know SAM does not win every cell.

**The "why" is supported less directly than the "what".** The flatness evidence is one Hessian-spectrum comparison (Figure 3, left) from a single model without batch norm. Appendix Table 11 is a more interesting data point: using more inner ascent steps finds a much larger estimated sharpness on CIFAR-10 (0.17 rising to 2.26) while test error stays at about 2.77 for 1, 2, 3 and 5 steps. If sharpness reduction were the mechanism, how the sharpness estimate changes would be expected to move test error. The appendix also reports that including the second-order terms hurts performance and says the reason is "unclear".

**Label-noise comparison (Table 4)** mixes numbers copied from the literature with numbers from the authors' own implementation, and the Mixup baselines were tuned over four values of alpha, so the comparison is not fully controlled.

## 7. Candidate alternative hypotheses (not yet chosen)

We must pick one by about Oct 27 so the one-page check-in summary can state it. Each candidate below is consistent with the paper's reported numbers and makes a different prediction from the paper's explanation. Section 6 of the assignment requires a prediction table before we run anything.

**H1: the gain depends on small per-device batches, not on sharpness.** In the paper's runs, perturbation and batch norm both operated on 32-example sub-batches. The benefit may come from that small-batch noise rather than from flatness. If the paper is right, SAM at m = 256 should still beat SGD by a similar margin. If H1 is right, the gain shrinks at m = 256 and returns when we emulate m = 32, and SGD trained with 32-example batch norm statistics should close part of the gap. This is the most natural hypothesis because our single-GPU setup already forces the question, and it can be tested with a code patch that splits each batch into 8 chunks.

**H2: the gain is partly a tuning or compute asymmetry.** If SGD received the same extra search budget as SAM (for example, a rho-sized grid over something SGD can use, such as gradient clipping or label smoothing), the 0.8 to 2.8 point gaps in Table 1 shrink. The paper's evidence cannot rule this out, and we can test it cheaply on CIFAR-10 with a small grid.

**H3: the benefit comes from an implicit gradient-norm or effective-step effect, not flatness.** SAM applies the gradient from a perturbed point, which acts like a gradient-norm regularizer. If that is the mechanism, an explicit gradient-norm penalty (or a different regularizer that does not reduce sharpness) should reproduce the gain. If the paper is right, matching the gain without reducing sharpness should not be possible. This is the most direct attack on the "why" but needs more implementation work.

For each hypothesis we will write the prediction table (paper's explanation vs ours) before running anything, and change only one thing relative to the paper's setup.

## 8. Plan: compute budget and run matrix

All timings below are **estimates** extrapolated from the measured T4 speeds, assuming the A100 is 4 to 6 times faster. They must be replaced with measured A100 numbers in the first hour of A100 access.

| Run (WRN-28-10, CIFAR-10, batch 256, lr 0.1, wd 0.0005, basic aug) | Estimated A100 time |
|---|---|
| SAM, rho 0.05, 200 epochs, one seed | ~4 to 6 h |
| SGD, 200 epochs, one seed | ~2.5 to 3.5 h |
| SGD, 400 epochs, one seed (the paper's doubled-epoch baseline) | ~5 to 7 h |

One full comparison (SAM, SGD-200, SGD-400) is roughly 12 to 16 A100-hours. The course guideline allows about 35 hours for replicating the key figure and 35 for our own experiment. Three seeds of everything would take about 36 to 48 hours, which exceeds the replication budget, so the provisional plan is: three seeds each of SAM and SGD-200 (about 20 to 30 hours), one seed of SGD-400, CIFAR-10 only, and CIFAR-100 only if budget remains. The first target is the single Table 1 row: SAM 2.7 +/- 0.1 vs SGD 3.5 +/- 0.1 test error (CIFAR-100 in the same row: 16.5 vs 18.8). If our single-seed runs land within a few tenths of those numbers, the setup is trustworthy. If not, that is reportable, and we investigate the m = 256 effect first. Debug on 2-epoch runs, time one epoch before launching anything long, launch long runs inside `tmux`, write to persistent storage, and use `--evaluate_every 10` to avoid per-epoch evaluation overhead.

## 9. Next steps and owners

| # | Task | Owner | Proposed due | Status |
|---|---|---|---|---|
| 1 | Every member runs the shared Colab notebook once and reports GPU type and epoch times | All | Oct 22 | Not started |
| 2 | Rebuild the environment on the course A100 from the `NOTES.md` recipe; copy the cached dataset; time one SAM and one SGD epoch | Code lead (@Siddharth-magesh) | Oct 23 | Not started |
| 3 | Each member reads the paper independently (three passes) and writes the central claim in their own words | All | Oct 24 | Not started |
| 4 | Team meeting: agree the central claim, list weak points, choose the alternative hypothesis | All | Oct 26 | Not started |
| 5 | Verify the seeding findings (read `load_model.py`, `dataset_source.py` fully); patch seeding as its own commit | TBD | Oct 26 | Not started |
| 6 | Launch the first 200-epoch SAM and SGD runs on the A100 | Code lead | Oct 26 | Not started |
| 7 | Write the prediction table for the chosen hypothesis | TBD | Oct 28 | Not started |
| 8 | Write and submit the one-page check-in summary (claim, key experiments, draft hypothesis, code status) | TBD | Oct 29 | Not started |
| 9 | If H1: design the m = 32 emulation patch (8 chunks per batch) | TBD | Nov 3 | Not started |
| 10 | Full report draft for the Nov 20 review | All | Nov 18 | Not started |

Owners marked TBD should be filled in at the next meeting. Each task should have exactly one owner, even if several people help.

## 10. Risks and open questions

**A100 environment compatibility.** The old CUDA 11.0 `jaxlib` should support the A100's architecture, but we have only tested on a T4. If it fails on the A100, the fallback is the PyTorch implementation by David Samuel (`davda54/sam`), which the paper's authors acknowledge. It is third-party, so we would have to tell the TAs and say so in the report.

**Budget overrun.** Section 8's times are guesses. A slower-than-expected A100 epoch changes the plan, so task 2 must happen early.

**Colab disconnects.** Long runs on Colab are unreliable. Real runs belong on the A100, with outputs on persistent storage. The training code resumes from checkpoints, which helps, but a stale checkpoint will also silently resume the wrong run, so every experiment needs a fresh output directory.

**Open questions.** Is the seeding behavior exactly as inferred? Does the doubled-epoch SGD baseline matter for our conclusions (we run it to find out)? Which hypothesis do we commit to? Is the exact cuDNN version on disk recorded in `NOTES.md`?

## 11. Housekeeping and course rules

The report is 4 pages excluding references and appendix, in a standard conference template. The appendix must contain extra plots, hyperparameters, failed attempts, a link to our code, a short statement of who did what, and a statement of how AI tools were used. We are allowed to use AI tools to understand the paper, debug code, and polish writing, but not to generate the analysis or conclusions, and every member must be able to explain the report if the TAs ask. To make the appendix statement easy, **each member should keep a running note of where they used AI help** (which task, which tool), added to the bottom of this file. Report failed runs and changed plans honestly, and do not adjust the experiment after seeing results to make it say something. Check-in and report-review milestones are not graded separately, but groups that skip them rarely do well, so everyone should come prepared.

## 12. Changelog

- **2026-10-20**: Tracker created. Environment working on Colab T4, SAM and SGD smoke tests passed, code findings documented, shared notebook prepared. A100 setup and first real reproduction not yet started.

## 13. AI-usage log (for the appendix statement)

- *(add entries: date, who, tool, what it was used for)*