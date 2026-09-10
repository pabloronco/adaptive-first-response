# Team Git / VS Code Workflow

Status: CURRENT DEFAULT for the rehearsal implementation.

This repository is the canonical codebase. Local VS Code folders are working copies, not separate project sources of truth.

## Core rule

Do not develop substantial features directly on `main`.

For each task:

1. Update local `main`.
2. Create a task branch.
3. Work locally in VS Code.
4. Run tests.
5. Commit and push the branch.
6. Open a pull request into `main`.
7. Review interface-sensitive changes before merge.

## Daily local sync

```bash
git switch main
git pull origin main
```

Then create a branch:

```bash
git switch -c <owner>/<short-task-name>
```

Examples:

```text
pablo/environment-skeleton
fede/environment-tests
demu/rl-prototype-import
```

## Before pushing

```bash
git status
git diff
pytest
```

Then:

```bash
git add <files>
git commit -m "<clear message>"
git push -u origin <branch-name>
```

Open a PR into `main`.

## Existing local prototypes

Existing rehearsal/prototype code must not be copied directly into `main`.

Import it on a dedicated branch, preserving provenance. The preferred branch for Demu's existing learned-policy prototype is:

```text
demu/rl-prototype-import
```

The prototype may be adapted to the frozen interfaces, but it must not silently redefine `GraphState`, `MissionAction`, hidden-truth boundaries, reward/evaluation semantics, or environment ownership.

Useful components can then be reviewed and merged incrementally.

## VS Code

Each collaborator should open the cloned repository folder itself as the VS Code workspace. Do not work from unrelated folders and manually copy files back and forth.

Recommended local layout:

```text
~/Documents/adaptive-first-response/
```

Use a repository-local Python environment:

```text
.venv/
```

`.venv` is ignored by Git and must not be committed.

## Merge discipline

Changes requiring team review before merge:

- `GraphState` schema
- `MissionAction` schema
- planner interface
- hidden/public state boundary
- action-space granularity
- reward components
- world-model train/holdout assignment
- evaluation metrics

Normal implementation details can be merged after tests and one lightweight review.
