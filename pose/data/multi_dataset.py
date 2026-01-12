from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import bisect

from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    dataset: Dataset
    loss_weight: float = 1.0
    name: Optional[str] = None


class WeightedConcatDataset(Dataset):
    """Concatenate multiple datasets and attach per-sample dataset metadata.

    Each sample dict returned by the underlying datasets is augmented with:
    - sample["dataset_weight"]: float (importance weight for loss)
    - sample["meta"]["dataset_idx"]: int
    - sample["meta"]["dataset_name"]: str (if provided)

    This keeps downstream training code model-agnostic while enabling
    multi-dataset training with importance weights.
    """

    def __init__(self, specs: List[DatasetSpec]):
        super().__init__()
        if not specs:
            raise ValueError("WeightedConcatDataset requires at least one dataset")

        self.specs = list(specs)
        self.datasets = [s.dataset for s in self.specs]

        # precompute cumulative sizes for fast index routing
        self.cumulative_sizes: List[int] = []
        running = 0
        for ds in self.datasets:
            running += len(ds)
            self.cumulative_sizes.append(running)

        # proxy commonly used attributes from the first dataset
        ref = self.datasets[0]
        self.num_keypoints = getattr(ref, "num_keypoints", None)
        self.input_size = getattr(ref, "input_size", None)
        self.heatmap_size = getattr(ref, "heatmap_size", None)
        self.depth_range = getattr(ref, "depth_range", None)
        self.depth_mean = getattr(ref, "depth_mean", None)

        # sanity check: num_keypoints consistency when available
        for i, ds in enumerate(self.datasets[1:], start=1):
            nk = getattr(ds, "num_keypoints", None)
            if self.num_keypoints is not None and nk is not None and nk != self.num_keypoints:
                raise ValueError(
                    f"All datasets must have same num_keypoints; dataset[0]={self.num_keypoints} vs dataset[{i}]={nk}"
                )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - prev_cum
        return dataset_idx, sample_idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self._locate(idx)
        spec = self.specs[dataset_idx]
        sample = self.datasets[dataset_idx][sample_idx]

        # enforce dict-like samples (current codebase uses dict batches)
        if not isinstance(sample, dict):
            raise TypeError(
                "WeightedConcatDataset expects underlying datasets to return dict samples; "
                f"got {type(sample)} from dataset {dataset_idx}."
            )

        # Attach dataset weight (as float; default collate -> tensor(B,))
        sample["dataset_weight"] = float(spec.loss_weight)

        meta = sample.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["dataset_idx"] = int(dataset_idx)
        if spec.name is not None:
            meta["dataset_name"] = str(spec.name)
        sample["meta"] = meta

        return sample
