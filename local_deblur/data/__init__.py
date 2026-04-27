"""Data contracts, datasets, and synthetic blur helpers."""

from .datasets import DryRunDeblurDataset, ManifestDeblurDataset, load_dataset
from .synthetic_blur import make_dry_run_sample, write_dry_run_artifacts
from .types import LocalDeblurRecord, LocalDeblurSample

try:
    from .tensor_dataset import TensorManifestDeblurDataset, deterministic_split_indices, sample_to_tensors
except ImportError as exc:  # Keep non-torch fallback data paths importable.
    if getattr(exc, "name", "") != "torch":
        raise
    TensorManifestDeblurDataset = None
    deterministic_split_indices = None
    sample_to_tensors = None

__all__ = [
    "DryRunDeblurDataset",
    "deterministic_split_indices",
    "LocalDeblurRecord",
    "LocalDeblurSample",
    "ManifestDeblurDataset",
    "TensorManifestDeblurDataset",
    "load_dataset",
    "make_dry_run_sample",
    "sample_to_tensors",
    "write_dry_run_artifacts",
]
