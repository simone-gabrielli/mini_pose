import cv2
from typing import List, Tuple


class HaarFaceDetector:
    """Simple OpenCV Haar cascade face detector.

    Detects faces and returns bounding boxes as (x, y, w, h) in the
    coordinate space of the input image.
    """

    def __init__(self, cascade_path: str | None = None, scale_factor: float = 1.1, min_neighbors: int = 5):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image_bgr) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a BGR image.

        Returns a list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
