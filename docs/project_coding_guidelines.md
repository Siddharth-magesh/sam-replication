# Project coding guidelines (for AI coding sessions)

Repo: `Siddharth-magesh/sam-replication`, a fork of `google-research/sam`. Goal: replicate SAM (Foret et al., ICLR 2021) on CIFAR-10 and test an alternative hypothesis. Read `docs/PROJECT_TRACKING.md` and `docs/NOTES.md` before doing anything. Follow these rules exactly; ask the user if a rule conflicts with the task.

## Branches and publication

- User-directed workflow (2026-11-17): keep `repro` as the sole branch and GitHub default. Work on the existing `repro` checkout; do not create feature branches or PRs unless the user requests them again.
- The user explicitly authorized consolidating the prepared Jetstream work into `repro` and deleting the other branches after their commits are preserved there. This supersedes the earlier feature-branch/PR-only workflow.
- Preserve existing work and executed result notebooks. Do not force-push or rewrite history.
- Commit logical changes with imperative messages. Changes to the authors' code stay isolated and documented.
- Push when authorized by the user; never include credentials, datasets or raw run artifacts.

## Layout (put files only here)
```
sam_jax/        authors' code. Change as little as possible; patches isolated and documented
requirements.txt  original pins. Do not edit; record env fixes in docs/NOTES.md
experiments/    launch scripts, one per run config
analysis/       scripts that read run outputs and produce tables/plots
results/        small final results only (csv, json, png)
notebooks/      Colab notebook, outputs cleared before commit
docs/           ALL markdown except the root README.md
```
Do not move, rename or restructure `sam_jax/`. The package must still run as `python -m sam.sam_jax.train` from the parent of a folder named `sam`.

## Markdown rules
- Only these md files exist: root `README.md`, `docs/PROJECT_TRACKING.md`, `docs/NOTES.md`, this file. Update them; do not create new ones.
- A new md file only if the user asks. Then it goes in `docs/` (or `docs/<topic>/`) with a descriptive snake_case name.
- User-directed organization (November 19): run instructions and experiment handoffs belong beside their scripts under `experiments/<config>/`; completed analyses live under `docs/results/`, historical diagnostics under `docs/archive/`. Keep the core notes/tracking/guidelines in docs and update the docs index and links when moving files.
- Links to code or results in docs point to the `repro` branch (`.../blob/repro/<path>`), never `main`.

## Never commit
Run outputs, logs, checkpoints, TensorBoard events, datasets (`tensorflow_datasets`, tfrecords), `__pycache__`/`.pyc`, notebook outputs, credentials or tokens, files over a few MB. No test files (pytest or otherwise), scratch scripts, or temporary md files. Keep scratch work outside the repo and delete it. Check `git status` and `git diff --staged` before every commit. Keep `.gitignore` covering all of the above.

## Running experiments
- Environment is fixed: Python 3.8, jax 0.2.6, jaxlib 0.1.57+cuda110, flax 0.2.2, cuDNN 8.x. No upgrades or new dependencies without asking. New code must run on this stack.
- Always pass explicitly: `--batch_size 256 --weight_decay 0.0005 --learning_rate 0.1` (code defaults are 128 / 0.001). SGD baseline is `--sam_rho 0.0`.
- Use a fresh `--output_dir` for every experiment (training resumes from any existing checkpoint). Write runs outside the repo.
- Never start a GPU run longer than 1 hour without the user's approval. Time one epoch first and log GPU-hours used.
- Report final-epoch test error, not the best epoch. Report only numbers from actual logs; label estimates as estimates; never edit, hide or re-run results to fit a story. Record failed runs.

## Tracking (required in every PR)
- Update `docs/PROJECT_TRACKING.md`: the task status table (Section 9), one dated line in the changelog (Section 12), and one entry in the AI-usage log (Section 13: date, tool, what it was used for).
- Put new technical findings, deviations from the paper, and setup changes in `docs/NOTES.md`.
- Mark unconfirmed claims **(to verify)**; do not state guesses as facts.
