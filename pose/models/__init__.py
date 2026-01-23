from .stacked_hourglass import StackedHourglass  # Expose the StackedHourglass model
from .fan2d import Fan2D  # Expose the new Fan2D model
from .mobilenet_pose import MobileNetPose  # Expose the MobileNetPose model
from .glasses_pose_regressor import GlassesPoseRegressor  # Direct pose regressor for glasses
from .lotr import LOTR, LOTRLight  # LOTR: Localization Transformer for landmarks

# Detectors also register via MODEL_REGISTRY; import to ensure availability during training.
from pose.detectors.face_detector import TinyFaceDetector  # noqa: F401
