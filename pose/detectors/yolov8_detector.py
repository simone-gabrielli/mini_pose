from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single detection result.

    Coordinates are in absolute pixel coords relative to the input image.
    """

    x: int
    y: int
    w: int
    h: int
    score: float
    class_id: int


ClassFilter = Optional[Sequence[Union[int, str]]]


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401

        return
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics YOLOv8 is not installed. Install it with: pip install ultralytics\n"
            "Note: this will pull additional dependencies (matplotlib, etc.)."
        ) from e


def _coerce_device(device: str | int | None) -> str | int | None:
    if device is None:
        return None
    # Ultralytics accepts: 'cpu', 'cuda', '0', 0, 'cuda:0'
    if isinstance(device, int):
        return device
    d = str(device).strip()
    if d == "":
        return None
    return d


class YOLOv8Detector:
    """Thin wrapper around Ultralytics YOLOv8 for bbox detection.

    This is intended for *inference-time* use inside this repo.

    The returned format matches the detector convention used by
    `scripts/infer_video.py`: list of (x, y, w, h, score).
    """

    def __init__(
        self,
        weights_path: str,
        *,
        device: str | int | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: ClassFilter = None,
    ) -> None:
        _ensure_ultralytics()
        from ultralytics import YOLO

        self.weights_path = str(weights_path)
        self.device = _coerce_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self._raw_class_filter: ClassFilter = list(classes) if classes is not None else None

        self.model = YOLO(self.weights_path)
        self.names = getattr(self.model, "names", None)
        self._class_ids = self._resolve_class_ids(self._raw_class_filter)

    def _resolve_class_ids(self, classes: ClassFilter) -> Optional[List[int]]:
        if classes is None:
            return None

        out: List[int] = []
        names = self.names
        names_by_lower = None
        if isinstance(names, dict):
            names_by_lower = {str(v).lower(): int(k) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            names_by_lower = {str(v).lower(): int(i) for i, v in enumerate(names)}

        for c in classes:
            if isinstance(c, int):
                out.append(int(c))
                continue

            s = str(c).strip()
            if s == "":
                continue
            if s.isdigit():
                out.append(int(s))
                continue

            if names_by_lower is None:
                raise ValueError(
                    "Class-name filtering requires model.names to be available. "
                    "Pass numeric class ids instead (e.g. --yolo-classes 0)."
                )

            key = s.lower()
            if key not in names_by_lower:
                known = ", ".join(sorted(names_by_lower.keys())[:50])
                raise ValueError(f"Unknown class '{s}'. Known classes: {known}")
            out.append(int(names_by_lower[key]))

        # de-dup while preserving order
        seen = set()
        uniq: List[int] = []
        for i in out:
            if i in seen:
                continue
            seen.add(i)
            uniq.append(i)
        return uniq

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        dets = self.detect_with_class(img_bgr)
        return [(d.x, d.y, d.w, d.h, d.score) for d in dets]

    def detect_with_class(self, img_bgr: np.ndarray) -> List[Detection]:
        if img_bgr is None or not hasattr(img_bgr, "shape"):
            return []

        H, W = img_bgr.shape[:2]
        if H <= 0 or W <= 0:
            return []

        # Ultralytics expects BGR uint8 numpy (OpenCV) just fine.
        res = self.model.predict(
            img_bgr,
            imgsz=int(self.imgsz),
            conf=float(self.conf),
            iou=float(self.iou),
            device=self.device,
            verbose=False,
        )
        if not res:
            return []

        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        try:
            xyxy = boxes.xyxy
            confs = boxes.conf
            clses = boxes.cls
        except Exception:
            return []

        if xyxy is None or confs is None or clses is None:
            return []

        xyxy_np = xyxy.detach().cpu().numpy().astype(np.float32)
        conf_np = confs.detach().cpu().numpy().astype(np.float32)
        cls_np = clses.detach().cpu().numpy().astype(np.float32)

        keep_ids = set(self._class_ids) if self._class_ids is not None else None

        out: List[Detection] = []
        for (x1, y1, x2, y2), sc, cl in zip(xyxy_np, conf_np, cls_np):
            class_id = int(cl)
            if keep_ids is not None and class_id not in keep_ids:
                continue

            x1i = int(max(0, min(W - 1, round(float(x1)))))
            y1i = int(max(0, min(H - 1, round(float(y1)))))
            x2i = int(max(0, min(W - 1, round(float(x2)))))
            y2i = int(max(0, min(H - 1, round(float(y2)))))

            if x2i <= x1i or y2i <= y1i:
                continue

            out.append(
                Detection(
                    x=x1i,
                    y=y1i,
                    w=int(x2i - x1i),
                    h=int(y2i - y1i),
                    score=float(sc),
                    class_id=class_id,
                )
            )

        out.sort(key=lambda d: d.score, reverse=True)
        return out
