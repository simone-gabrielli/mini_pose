"""Convert COCO detection annotations to a YOLOv8-compatible dataset layout.

Creates a dataset directory like:
  out_root/
    data.yaml
    images/train/...   (hardlinks or copies)
    images/val/...
    labels/train/...   (YOLO .txt labels)
    labels/val/...

This is designed for training a YOLOv8 detector (e.g. glasses detection) while
keeping your source images/COCO JSONs intact.

Example:
  python tools/coco_to_yolo.py \
    --train-json datasets/my_ds/annotations/train.json \
    --val-json datasets/my_ds/annotations/val.json \
    --image-root datasets/my_ds/images \
    --out-root datasets/my_ds_yolo \
    --categories glasses

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


def _parse_csv(s: str | None) -> Optional[List[str]]:
    if not s:
        return None
    items = [p.strip() for p in str(s).split(",")]
    items = [p for p in items if p]
    return items or None


def _load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_categories(coco: dict, categories: Optional[Sequence[str]]) -> Tuple[List[dict], Dict[int, int]]:
    all_cats = coco.get("categories", []) or []
    if not isinstance(all_cats, list) or not all_cats:
        raise ValueError("COCO JSON has no 'categories' list")

    if not categories:
        selected = list(all_cats)
    else:
        wanted = []
        for c in categories:
            cs = str(c).strip()
            if cs == "":
                continue
            wanted.append(cs)

        selected = []
        by_id = {int(cat["id"]): cat for cat in all_cats if "id" in cat}
        by_name = {str(cat.get("name", "")).lower(): cat for cat in all_cats}

        for w in wanted:
            if w.isdigit() and int(w) in by_id:
                selected.append(by_id[int(w)])
            else:
                k = w.lower()
                if k not in by_name:
                    known = ", ".join(sorted(by_name.keys())[:50])
                    raise ValueError(f"Unknown category '{w}'. Known category names: {known}")
                selected.append(by_name[k])

        # de-dup while preserving order
        seen = set()
        uniq = []
        for cat in selected:
            cid = int(cat["id"])
            if cid in seen:
                continue
            seen.add(cid)
            uniq.append(cat)
        selected = uniq

    cat_id_to_idx: Dict[int, int] = {}
    for i, cat in enumerate(selected):
        cat_id_to_idx[int(cat["id"])] = i

    return selected, cat_id_to_idx


def _iter_coco_images(coco: dict) -> Iterable[CocoImage]:
    for img in coco.get("images", []) or []:
        try:
            yield CocoImage(
                id=int(img["id"]),
                file_name=str(img["file_name"]),
                width=int(img.get("width") or 0),
                height=int(img.get("height") or 0),
            )
        except Exception:
            continue


def _safe_link_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if copy_images:
        shutil.copy2(str(src), str(dst))
        return

    # Prefer hardlinks (fast, no admin). Fall back to copy.
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def _convert_split(
    *,
    split_name: str,
    coco_json: str,
    image_root: str,
    out_root: str,
    cat_id_to_idx: Dict[int, int],
    copy_images: bool,
    write_empty_labels: bool,
) -> Tuple[int, int]:
    coco = _load_coco(coco_json)
    images = list(_iter_coco_images(coco))

    ann_by_img: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []) or []:
        try:
            img_id = int(ann.get("image_id"))
        except Exception:
            continue
        ann_by_img.setdefault(img_id, []).append(ann)

    out_root_p = Path(out_root)
    out_images = out_root_p / "images" / split_name
    out_labels = out_root_p / "labels" / split_name

    img_root_p = Path(image_root) if image_root else Path(coco_json).parent

    n_images = 0
    n_labels = 0

    for im in images:
        if im.width <= 0 or im.height <= 0:
            # YOLO normalization needs image size.
            # If your COCO JSON doesn't include it, you should regenerate it with width/height.
            continue

        src_img = img_root_p / im.file_name
        if not src_img.exists():
            # Try relative to JSON folder as a fallback.
            alt = Path(coco_json).parent / im.file_name
            if alt.exists():
                src_img = alt
            else:
                continue

        dst_img = out_images / im.file_name
        _safe_link_or_copy(src_img, dst_img, copy_images=copy_images)

        rel = Path(im.file_name)
        dst_lbl = (out_labels / rel).with_suffix(".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for ann in ann_by_img.get(im.id, []):
            try:
                cid = int(ann.get("category_id"))
            except Exception:
                continue
            if cid not in cat_id_to_idx:
                continue

            bbox = ann.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 1e-3 or h <= 1e-3:
                continue

            # COCO bbox is x,y,w,h in pixels.
            xc = x + 0.5 * w
            yc = y + 0.5 * h

            xc_n = xc / float(im.width)
            yc_n = yc / float(im.height)
            w_n = w / float(im.width)
            h_n = h / float(im.height)

            # clamp to [0,1] to be safe
            xc_n = max(0.0, min(1.0, xc_n))
            yc_n = max(0.0, min(1.0, yc_n))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))

            cls_idx = cat_id_to_idx[cid]
            lines.append(f"{cls_idx} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}")

        if lines or write_empty_labels:
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_labels += 1

        n_images += 1

    return n_images, n_labels


def _write_data_yaml(out_root: str, class_names: List[str]) -> Path:
    # Avoid adding PyYAML dependency here; YAML is a superset and this is simple.
    out_root_p = Path(out_root)
    path = out_root_p / "data.yaml"

    lines = []
    lines.append(f"path: {out_root_p.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for i, name in enumerate(class_names):
        safe = str(name).replace("\"", "")
        lines.append(f"  {i}: {safe}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True, help="COCO train JSON")
    ap.add_argument("--val-json", required=True, help="COCO val JSON")
    ap.add_argument("--image-root", required=True, help="Root directory for images referenced by file_name")
    ap.add_argument("--out-root", required=True, help="Output YOLO dataset root")
    ap.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of category names/ids to export (e.g. 'glasses' or '1'). Default: all.",
    )
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of hardlinking them")
    ap.add_argument(
        "--write-empty-labels",
        action="store_true",
        help="Write empty .txt files for images with no objects (optional).",
    )

    args = ap.parse_args()

    train_json = str(args.train_json)
    val_json = str(args.val_json)
    image_root = str(args.image_root)
    out_root = str(args.out_root)

    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    # Resolve categories using TRAIN json as the source of truth
    coco_train = _load_coco(train_json)
    cats, cat_id_to_idx = _select_categories(coco_train, _parse_csv(args.categories))
    class_names = [str(c.get("name", f"cat_{c.get('id')}")) for c in cats]

    n_tr_img, n_tr_lbl = _convert_split(
        split_name="train",
        coco_json=train_json,
        image_root=image_root,
        out_root=out_root,
        cat_id_to_idx=cat_id_to_idx,
        copy_images=bool(args.copy_images),
        write_empty_labels=bool(args.write_empty_labels),
    )
    n_va_img, n_va_lbl = _convert_split(
        split_name="val",
        coco_json=val_json,
        image_root=image_root,
        out_root=out_root,
        cat_id_to_idx=cat_id_to_idx,
        copy_images=bool(args.copy_images),
        write_empty_labels=bool(args.write_empty_labels),
    )

    data_yaml = _write_data_yaml(out_root, class_names)

    print(f"[coco_to_yolo] Wrote: {data_yaml}")
    print(f"[coco_to_yolo] classes: {class_names}")
    print(f"[coco_to_yolo] train: images={n_tr_img} labels={n_tr_lbl}")
    print(f"[coco_to_yolo] val  : images={n_va_img} labels={n_va_lbl}")


if __name__ == "__main__":
    main()
