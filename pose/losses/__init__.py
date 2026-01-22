from .heatmap_mse import HeatmapMSELoss  # noqa: F401
from .depth_mse import DepthMSELoss  # noqa: F401
from .heatmap_3d_mse import Heatmap3DMSELoss  # noqa: F401
from .reprojection_loss import ReprojectionLoss  # noqa: F401

# Single-box detector loss (TinyFaceDetector, etc.)
from .bbox_detector import BBoxDetectorLoss  # noqa: F401

# LOTR-style coordinate regression losses
from .smooth_wing import (  # noqa: F401
    WingLoss,
    SmoothWingLoss,
    AdaptiveWingLoss,
    LandmarkL1Loss,
    LandmarkMSELoss,
    LandmarkSmoothL1Loss,
    NMELoss,
    CombinedLandmarkLoss,
)