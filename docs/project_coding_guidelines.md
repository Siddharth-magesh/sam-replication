# Project coding guidelines (for AI coding sessions)

Repo: `Siddharth-magesh/sam-replication`, a fork of `google-research/sam`. Goal: replicate SAM (Foret et al., ICLR 2021) on CIFAR-10 and test an alternative hypothesis. Read `docs/PROJECT_TRACKING.md` and `docs/NOTES.md` before doing anything. Follow these rules exactly; ask the user if a rule conflicts with the task.

## Branches and PRs
- `main` is never touched: no commits, no merges, no upstream syncs.
- `repro` is the integration branch. Never commit or push to it directly.
- For every task: `git checkout repro && git pull`, then `git checkout -b <type>/<short-name>` (`feat/`, `fix/`, `docs/`, `patch/`).
- Open the PR against `repro` in **our fork**, never against upstream:
  `gh pr create --repo Siddharth-magesh/sam-replication --base repro`
- Do not merge your own PR. No force-push, no history rewriting.
- One logical change per commit, imperative message (`patch: seed weight init from run_seed`). Changes to the authors' code go in their own commits.

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