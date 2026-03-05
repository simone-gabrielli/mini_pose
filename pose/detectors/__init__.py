from .face_detector import TinyFaceDetector

# Optional dependency: Ultralytics YOLOv8. This is used for glasses/bbox detection.
# We import it lazily to keep the base install lightweight.
try:  # pragma: no cover
    from .yolov8_detector import YOLOv8Detector  # noqa: F401
except Exception:
    YOLOv8Detector = None  # type: ignore
