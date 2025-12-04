import torch
import torch.nn as nn
import torchvision
from pose.registry import register_model


@register_model("tiny_face")
class TinyFaceDetector(nn.Module):
    """A lightweight face detector that regresses a single bounding box + confidence

    - Input: RGB image tensor, shape (B,3,H,W) expected in [0,1]
    - Output: tensor (B,5) -> [conf_logit, x1, y1, x2, y2] with bbox coords normalized [0..1]
    """

    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        # MobileNetV2 features as lightweight backbone
        # Use the new `weights=` enum when available to avoid deprecation warnings.
        try:
            weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.mobilenet_v2(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
        # keep feature extractor (all features up to classifier)
        self.features = backbone.features

        # global pooling + small head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.last_channel, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 5),
        )

    def forward(self, x):
        # x assumed normalized already
        feat = self.features(x)
        pooled = self.pool(feat)
        out = self.fc(pooled)
        # out[:,0] is raw logit for confidence
        # out[:,1:5] are raw values; we'll sigmoid those during loss/prediction
        return out

    @torch.no_grad()
    def predict(self, x, conf_th=0.3):
        """Run forward and return (conf, bbox) in absolute coords for each image.

        Returns list of dicts: {'conf': float, 'bbox': [x1,y1,x2,y2]} with absolute pixel coords
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        out = self.forward(x)
        logits = out[:, 0]
        conf = torch.sigmoid(logits)
        bbox = torch.sigmoid(out[:, 1:5])

        results = []
        _, _, H, W = x.shape
        for c, b in zip(conf.cpu(), bbox.cpu()):
            c_val = float(c.item())
            b = b.numpy()
            # convert normalized x1,y1,x2,y2 to pixel coords
            x1 = float(b[0] * W)
            y1 = float(b[1] * H)
            x2 = float(b[2] * W)
            y2 = float(b[3] * H)
            if c_val >= conf_th:
                results.append({"conf": c_val, "bbox": [x1, y1, x2, y2]})
            else:
                results.append({"conf": c_val, "bbox": None})
        return results


def load_tiny_face_detector(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyFaceDetector(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
