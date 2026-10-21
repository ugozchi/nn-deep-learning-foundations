# Contributing (final project)

We are three on the final project. `main` must always run: **every change goes through a pull request**.

## Workflow

1. **Pick an issue** and assign yourself. No issue yet? Open one first (templates: task, experiment, bug).
2. **Branch from an up-to-date `main`**, named `<type>/<issue>-<short-description>`:
   ```bash
   git switch main && git pull
   git switch -c feat/5-mlp-classifier
   ```
3. **Commit small, focused changes** using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(mlp): add 2-hidden-layer classifier
   fix(data): stratify the train/val split
   exp(resnet): compare frozen backbone and fine-tuning
   docs: record the subset build command
   ```
   Types: `feat`, `fix`, `exp` (experiment / results), `refactor`, `docs`, `chore`.
4. **Open a pull request** (a draft early is fine) and link the issue with `Closes #5`.
5. **One teammate reviews** before merging. Merges are **squash merges**, so `main` gets one clean commit per PR; the branch is deleted automatically.

## Rules of thumb

- **Shared code goes in `Final Project/src/`**, experiments in `Final Project/notebooks/`. A notebook has a single owner, to avoid merge conflicts in `.ipynb` files.
- **Restart & run all** before opening a PR, so the outputs in the notebook match the code.
- **Every number is measured**: fix the seed, keep metrics in `results/` (json/csv), save figures there too.
- **Never commit data or weights**: WikiArt images (licence), `*.pt` / `*.pth` checkpoints. They are git-ignored; keep it that way.
- Keep `main` green: if your branch is behind, `git pull --rebase origin main` before asking for a review.
