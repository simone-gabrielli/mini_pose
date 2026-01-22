
# mini-pose: A Minimal, Extensible Pose Estimation Framework

mini-pose is a **simple, modern, dependency-stable** framework for training and deploying human/facial keypoint detection models.

Unlike MMPose/Detectron2, mini-pose is:

- lightweight  
- pure PyTorch (no mmcv/MMEngine)  
- easy to extend  
- suitable for custom datasets  
- perfect for research, prototyping, embedded systems

The framework supports:

- **Stacked Hourglass** architecture  
- **COCO-style datasets** (e.g., 68-point face annotations)  
- **Albumentations-based augmentations**  
- **Heatmap generation**  
- **Flexible registry system**  
- **Multiple loss functions**  
- **PCK and NME metrics**  
- **Clean modular design**  

---

## 1. Folder Structure

```
mini_pose/
  pose/
    __init__.py
    config.py
    registry.py
    models/
      __init__.py
      base.py
      stacked_hourglass.py
    data/
      __init__.py
      dataset_coco.py
      heatmap.py
      albu_aug.py
    losses/
      __init__.py
      heatmap_mse.py
    engine/
      __init__.py
      trainer.py
      inference.py
  scripts/
    train.py
    infer.py
  configs/
    face68_hourglass.yaml
  requirements.txt
  README.md
```

---

## 2. Installation (Windows + Conda)

### 2.1 Create Conda Environment

```bash
conda create -n mini_pose python=3.10 -y
conda activate mini_pose
```

### 2.2 Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Install requirements

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 3. Quick Start

### Train a model:

```bash
python scripts/train.py --config configs/face68_hourglass.yaml
```

### Train a lightweight face bbox detector:

```bash
python scripts/train.py --config configs/face_mobilenet.yaml
```

This uses dataset type `coco_face` (single-face bbox regression with optional negative images) and loss `bbox_detector`.

### Train on multiple datasets (importance-weighted):

Use `data.train_datasets` to mix multiple COCO-style datasets during training.
Each dataset can have its own `loss_weight` to scale its contribution to the total loss.

See the full example config at configs/xreal_mobilenet_hmd_plus_air2.yaml.

### Run inference:

```bash
python scripts/infer.py --checkpoint work_dirs/face68_hourglass/best.pth                         --image test.jpg                         --num-keypoints 68
```

Outputs a `debug_out.png` with detected points.

---

## 4. How To Extend The Framework

Everything inside mini-pose uses **registries**, so you can easily plug in new modules. 

There are **two required steps** for every new module type (model, loss, dataset):

1. **Register the class** with a decorator in `pose/registry.py` (`@register_model`, `@register_loss`, `@register_dataset`).
2. **Make sure the module file is imported at startup** so the decorator actually runs and fills the registry.

If you skip step 2, you will see errors like `KeyError: 'stacked_hourglass'` or `KeyError: 'coco_keypoints'` because the registry is still empty for that key.

---

### 4.1 Extending Models

1. **Create the model file** in `pose/models/`:

```python
from pose.registry import register_model
from pose.models.base import PoseModel

@register_model("my_model")
class MyModel(PoseModel):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x):
        # must return (last_heatmap, list_of_all_heatmaps)
        return last_heatmap, [last_heatmap]
```

2. **Expose it in `pose/models/__init__.py`** so importing `pose.models` will register it:

```python
from .stacked_hourglass import StackedHourglass  # existing
from .my_model import MyModel                    # new
```

3. **Ensure models are imported in the trainer** (`pose/engine/trainer.py`):

```python
import pose.models    # noqa: F401  # ensures models register
```

4. **Use the model in the config**:

```yaml
model:
  name: my_model
  # other model-specific args...
```

---

### 4.2 Extending Losses

1. **Create the loss file** in `pose/losses/`:

```python
from pose.registry import register_loss
import torch.nn as nn

@register_loss("wing_loss")
class WingLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, preds, targets, visible=None):
        # preds, targets: (B, K, H, W)
        # visible: (B, K) or None
        return loss
```

2. **Expose it in `pose/losses/__init__.py`** so importing `pose.losses` will register it:

```python
from .heatmap_mse import HeatmapMSELoss  # existing
from .wing_loss import WingLoss          # new
```

3. **Ensure losses are imported in the trainer** (`pose/engine/trainer.py`):

```python
import pose.losses    # noqa: F401  # ensures losses register
```

4. **Use the loss in the config**:

```yaml
loss:
  name: wing_loss
```

If you forget step 2 or 3, you will get `KeyError: 'wing_loss'` when the trainer looks up `LOSS_REGISTRY[cfg["loss"]["name"]]`.

---

### 4.3 Extending Datasets

1. **Create the dataset file** in `pose/data/`:

```python
from torch.utils.data import Dataset
from pose.registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(Dataset):
    def __init__(self, json_path, image_root, input_size, heatmap_size, ...):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        # must return a dict with keys like:
        #   image: Tensor (C,H,W)
        #   keypoints: Tensor (K,3)
        #   heatmaps: Tensor (K,Hh,Wh)
        #   visible: Tensor (K)
        #   meta: dict
        return sample
```

2. **Expose it in `pose/data/__init__.py`** so importing `pose.data` will register it:

```python
from .dataset_coco import CocoKeypointsDataset   # existing
from .my_dataset import MyDataset                # new
```

3. **Ensure datasets are imported in the trainer** (`pose/engine/trainer.py`):

```python
import pose.data      # noqa: F401  # ensures datasets register
```

4. **Use the dataset in the config**:

```yaml
data:
  type: my_dataset
  train_json: ...
  val_json: ...
  image_root: ...
  input_size: [256, 256]
  heatmap_size: [64, 64]
```

If any of these imports are missing, you will see `KeyError: 'my_dataset'` when the trainer tries to access `DATASET_REGISTRY[ds_cfg["type"]]`.

---

### 4.4 Extending Augmentations

Modify `pose/data/albu_aug.py` and add any Albumentations transforms:

```
A.MotionBlur()
A.Perspective()
A.CoarseDropout()
A.ColorJitter()
...
```

---

### 4.5 Extending Metrics

Metrics live in `pose/engine/metrics.py`.

Add:

```python
def my_metric(preds, targets):
    ...
```

And call it inside validation.

---

## 5. Config Format

Example: `configs/face68_hourglass.yaml`

```yaml
experiment: face68_hourglass
seed: 42

data:
  type: coco_keypoints
  train_json: "datasets/face/annotations/train.json"
  val_json: "datasets/face/annotations/val.json"
  image_root: "datasets/face/images"
  input_size: [256, 256]
  heatmap_size: [64, 64]
  batch_size: 16
  num_workers: 4

  # Optional: augmentation config
  # aug:
  #   # Horizontal flip is a strong regularizer, but you MUST provide flip_pairs
  #   # so left/right landmark semantics stay correct.
  #   hflip_p: 0.5
  #   flip_pairs:  # 68-point example (0-indexed)
  #     - [0, 9]
  #     - [1, 10]
  #     - [2, 11]
  #     - [3, 12]
  #     - [4, 13]
  #     - [5, 14]
  #     - [6, 15]
  #     - [7, 16]
  #     - [8, 17]
  #     - [18, 33]
  #     - [19, 32]
  #     - [20, 31]
  #     - [21, 30]
  #     - [22, 29]
  #     - [23, 28]
  #     - [24, 27]
  #     - [34, 37]
  #     - [35, 38]
  #     - [40, 46]
  #     - [41, 47]
  #     - [42, 48]
  #     - [43, 49]
  #     - [44, 50]
  #     - [45, 51]
  #     - [52, 58]
  #     - [53, 59]
  #     - [54, 60]
  #     - [55, 61]
  #     - [56, 62]
  #     - [57, 63]
  #     - [64, 66]
  #     - [65, 67]

model:
  name: stacked_hourglass
  num_stacks: 2
  num_blocks: 1
  num_feats: 256

loss:
  name: heatmap_mse

train:
  epochs: 50
  lr: 0.00025
  weight_decay: 0.0001
  optimizer: adam
  lr_steps: [30, 40]
  lr_gamma: 0.1
  output_dir: "work_dirs/face68_hourglass"

  # --- Optional regularization bundle ---
  # Optimizer + schedule
  # optimizer: adamw
  # lr_schedule: cosine_warmup
  # warmup_epochs: 5
  # warmup_start_factor: 0.1
  # min_lr: 0.0

  # Training stability
  # grad_clip_norm: 1.0

  # Mixed precision (CUDA only)
  # amp:
  #   enabled: true
  #   dtype: fp16   # fp16 | bf16

  # EMA weights (often improves NME/PCK)
  # ema:
  #   enabled: true
  #   decay: 0.999
  #   update_every: 1
  #   eval: true
```

---

## 6. Dataset Preparation

mini-pose expects **COCO-format keypoints**:

```
dataset/
  images/
  annotations/
    train.json
    val.json
    test.json
```

You already have a DLIB → COCO converter.

---

## 7. Training Flow

1. Load config  
2. Albumentations augmentation  
3. Generate heatmaps  
4. Forward pass  
5. Compute multi-stack loss  
6. Validation (loss + PCK + NME)  
7. Save checkpoints  

---

## 8. Inference

```bash
python scripts/infer.py     --checkpoint work_dirs/face68_hourglass/best.pth     --image test.jpg
```

Result saved as `debug_out.png`.

---

## 9. Export to ONNX / TensorRT

```python
torch.onnx.export(model, dummy_input, "pose.onnx", opset=17)
```

---

## 10. Next Steps (Optional)

- Add sampling rebalancing for multi-dataset
- Add TensorBoard logging  
- Add Mixup/Mosaic keypoint augmentations  
