"""Path helpers shared by scripts and modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path | None, *, default: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    value = Path(path if path is not None else default if default is not None else ".")
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dated_result_dir(round_name: str, model: str, dataset: str, count: int | str) -> Path:
    """Create results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]."""
    date_part = datetime.now().strftime("%m%d")
    safe = "_".join(str(part).replace("/", "-").replace(" ", "-") for part in (round_name, model, dataset, count, date_part))
    base = PROJECT_ROOT / "results" / safe
    if not base.exists():
        base.mkdir(parents=True)
        return base

    suffix = datetime.now().strftime("%H%M")
    candidate = PROJECT_ROOT / "results" / f"{safe}_{suffix}"
    index = 1
    while candidate.exists():
        candidate = PROJECT_ROOT / "results" / f"{safe}_{suffix}_{index}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate
