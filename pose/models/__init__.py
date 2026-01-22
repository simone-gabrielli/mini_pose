from .stacked_hourglass import StackedHourglass  # Expose the StackedHourglass model
from .fan2d import Fan2D  # Expose the new Fan2D model
from .fan3d import Fan3D # Expose the Fan3D model
from .mobilenet_pose import MobileNetPose  # Expose the MobileNetPose model
from .mobilenet_pose_3d import MobileNetPose3D  # Expose the MobileNetPose3D model
from .glasses_pose_regressor import GlassesPoseRegressor  # Direct pose regressor for glasses
from .lotr import LOTR, LOTR3D, LOTRLight  # LOTR: Localization Transformer for landmarks

# Detectors also register via MODEL_REGISTRY; import to ensure availability during training.
from pose.detectors.face_detector import TinyFaceDetector  # noqa: F401
