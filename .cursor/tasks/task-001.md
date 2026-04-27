# Task 001: Project Foundation
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Create the minimal package skeleton, project metadata, dependency files, configuration conventions, and `.gitignore` entries needed for the local image deblurring workflow.

## Context
- Relevant files: [`/root/autodl-tmp/project/.gitignore`, `/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/pyproject.toml`, `/root/autodl-tmp/project/local_deblur/__init__.py`, `/root/autodl-tmp/project/local_deblur/config.py`, `/root/autodl-tmp/project/local_deblur/logging_utils.py`, `/root/autodl-tmp/project/local_deblur/paths.py`, `/root/autodl-tmp/project/configs/base.yaml`]
- Current state: Repository is mostly empty; `.gitignore` exists and currently ignores `.env`; `proposal.pdf` exists.
- Dependencies: none
- Domino assumptions: Build a minimal but runnable scaffold; datasets/checkpoints may be absent; smoke paths must avoid large downloads; keep edits scoped to project code and metadata.
- User decisions: Workspace is `/root/autodl-tmp/project`; follow `proposal.pdf`; update `.gitignore` to exclude `cache/`, `output/`, `results/`, `*.log`, `*.pyc`, and `*.tmp` while preserving existing entries.

## Instructions
1. Create a Python package named `local_deblur` with small shared modules for config loading, path handling, and logging.
2. Add `pyproject.toml` with basic project metadata and package discovery.
3. Add `requirements.txt` with practical dependencies for PyTorch/diffusers-compatible research code, image processing, metrics, YAML config, and progress/logging. Keep heavy dependencies optional in code where possible.
4. Add `configs/base.yaml` documenting default paths, dry-run behavior, image size `512`, target compute `A100 40GB`, and project modes.
5. Update `.gitignore` surgically, preserving existing lines and adding the required ignores.
6. Ensure modules import cleanly without datasets or checkpoints.

## Acceptance Criteria
- [ ] `local_deblur` package exists and can be imported.
- [ ] `requirements.txt` and `pyproject.toml` exist with coherent dependencies/metadata.
- [ ] `.gitignore` preserves `.env` and includes `cache/`, `output/`, `results/`, `*.log`, `*.pyc`, and `*.tmp`.
- [ ] `configs/base.yaml` captures default project settings and dry-run conventions.
- [ ] No training, inference, or evaluation implementation is added beyond shared foundation utilities.

## Output
- Modified files: [`/root/autodl-tmp/project/.gitignore`, `/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/pyproject.toml`, `/root/autodl-tmp/project/local_deblur/__init__.py`, `/root/autodl-tmp/project/local_deblur/config.py`, `/root/autodl-tmp/project/local_deblur/logging_utils.py`, `/root/autodl-tmp/project/local_deblur/paths.py`, `/root/autodl-tmp/project/configs/base.yaml`]
- Result summary: write under `## Result`

## Result
Files changed: `.gitignore`, `requirements.txt`, `pyproject.toml`, `local_deblur/config.py`, `local_deblur/logging_utils.py`, `local_deblur/paths.py`, `configs/base.yaml`.

Acceptance criteria: package foundation, project metadata, config/logging/path helpers, base config, and required ignore rules are implemented.

Constraint check: `.env` is preserved; generated caches, outputs, results, logs, pyc, and tmp files are ignored; no training, inference, or evaluation execution was added to this task scope.
