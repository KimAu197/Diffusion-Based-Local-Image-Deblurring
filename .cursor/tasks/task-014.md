# Task 014: ReLoBlur Collection And Manifest Conversion
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: blocked_

## Objective
Collect ReLoBlur if accessible, convert it into the project manifest format, and validate image/mask/target samples.

## Context
- Relevant files: [`/root/autodl-tmp/project/docs/data_status.md`, `/root/autodl-tmp/project/scripts/`, `/root/autodl-tmp/project/local_deblur/data/`, `/root/autodl-tmp/project/output/datasets/reloblur/`]
- Current state: ReLoBlur is not found locally. Public source is `https://github.com/LeiaLi/ReLoBlur`, with Google Drive and Baidu Cloud links described by the project.
- Dependencies: task-013
- Domino assumptions: ReLoBlur may require academic-use license acceptance, Google Drive quota, or Baidu credentials. If automated download is blocked, stop with one concrete question.
- User decisions: Collect ReLoBlur and use it as real data for posttraining/fine-tuning.

## Instructions
1. Check local paths for any existing ReLoBlur files before downloading.
2. Fetch/read the official ReLoBlur repository/project page to identify train/test/mask download links and license notes.
3. Attempt download only from official links or documented mirrors.
4. Convert the dataset into manifests with fields `Ib`, `M`, optional `S`, and `target`, preserving train/test split metadata.
5. Validate counts, dimensions, masks, and representative samples.
6. Update `docs/data_status.md` with source, license/access status, paths, counts, and manifest locations.
7. If download is blocked by authorization/quota/credentials, mark workflow blocked and ask one question.

## Acceptance Criteria
- [ ] ReLoBlur local status is known and documented.
- [ ] If accessible, train/test manifests are created and validated.
- [ ] If inaccessible, the exact blocked access requirement is documented.
- [ ] No unrelated dataset files are modified.

## Output
- Modified files: [`/root/autodl-tmp/project/docs/data_status.md`, `/root/autodl-tmp/project/scripts/`, `/root/autodl-tmp/project/output/datasets/reloblur/`, `/root/autodl-tmp/project/.cursor/tasks/task-014.md`]
- Result summary: write under `## Result`

## Result

Files changed:
- `scripts/prepare_reloblur_manifest.py`
- `docs/data_status.md`
- `.cursor/tasks/task-014.md`

Acceptance criteria:
- Local search found no ReLoBlur data under `/root`, `/root/autodl-tmp`, or `/autodl-pub/data`.
- Official sources were read from `LeiaLi/ReLoBlur`: Google Drive and Baidu links for train/test/masks, with academic/CC BY-NC-SA 4.0 license notes.
- Google Drive download attempt for the official test folder failed because the environment cannot reach `drive.google.com`: `Network is unreachable`.
- A converter script now exists for local ReLoBlur data and can generate `train_manifest.json` and `test_manifest.json` from `dataset/` and `masks/` roots.
- Converter script passed `py_compile` and CLI help smoke check.

Block status:
- ReLoBlur data collection is blocked by external data access, not by code. The project can proceed with SD + ControlNet COCO pretraining, but ReLoBlur posttraining needs one of:
  - a local uploaded/extracted ReLoBlur directory, or
  - working Google Drive access, or
  - Baidu Cloud download support/credentials.

Constraint check:
- Did not fabricate ReLoBlur data.
- Did not modify existing COCO synthetic data or baseline results.
- Prepared the manifest conversion path for immediate use once data is available.
