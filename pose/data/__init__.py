from .dataset_coco import CocoKeypointsDataset  # noqa: F401
from .dataset_coco_3d import CocoKeypoints3DDataset  # noqa: F401
from .dataset_face import CocoFaceDataset  # noqa: F401
from .multi_dataset import DatasetSpec, WeightedConcatDataset  # noqa: F401

__all__ = [
	"CocoKeypointsDataset",
	"CocoKeypoints3DDataset",
	"CocoFaceDataset",
	"DatasetSpec",
	"WeightedConcatDataset",
]


def get_cad_model_points():
	"""Placeholder helper for loading 3D glasses model points.

	This is a thin indirection so you can later swap in
	a proper loader (e.g. from a .ply/.obj or a JSON file)
	without touching the pose-regression head.
	"""
	raise NotImplementedError(
		"Implement get_cad_model_points() to return an array of shape (N, 3) "
		"with the 3D coordinates of the glasses model in object space."
	)

