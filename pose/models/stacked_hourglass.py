# pose/models/stacked_hourglass.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .base import PoseModel
from pose.registry import register_model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        return out


class Hourglass(nn.Module):
    def __init__(self, depth, num_features):
        super().__init__()
        self.depth = depth
        self.num_features = num_features

        self.upper = ResidualBlock(num_features, num_features)

        if depth > 1:
            self.lower1 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                ResidualBlock(num_features, num_features)
            )
            self.lower2 = Hourglass(depth - 1, num_features)
            self.lower3 = ResidualBlock(num_features, num_features)
        else:
            self.lower1 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                ResidualBlock(num_features, num_features)
            )
            self.lower2 = ResidualBlock(num_features, num_features)
            self.lower3 = ResidualBlock(num_features, num_features)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.upper(x)
        low1 = self.lower1(x)
        low2 = self.lower2(low1)
        low3 = self.lower3(low2)
        up2 = self.upsample(low3)
        return up1 + up2


@register_model("stacked_hourglass")
class StackedHourglass(PoseModel):
    def __init__(
        self,
        num_stacks: int,
        num_blocks: int,
        num_feats: int,
        num_keypoints: int,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_keypoints = num_keypoints

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, num_feats),
        )

        self.hourglasses = nn.ModuleList(
            [Hourglass(depth=4, num_features=num_feats) for _ in range(num_stacks)]
        )
        self.features = nn.ModuleList(
            [self._make_residual(num_blocks, num_feats) for _ in range(num_stacks)]
        )
        self.lin_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_feats, num_feats, 1, bias=False),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stacks)
            ]
        )
        self.pred_layers = nn.ModuleList(
            [nn.Conv2d(num_feats, num_keypoints, 1) for _ in range(num_stacks)]
        )

        # intermediate supervision
        self.merge_feats = nn.ModuleList(
            [nn.Conv2d(num_feats, num_feats, 1) for _ in range(num_stacks - 1)]
        )
        self.merge_preds = nn.ModuleList(
            [nn.Conv2d(num_keypoints, num_feats, 1) for _ in range(num_stacks - 1)]
        )

    def _make_residual(self, num_blocks, num_feats):
        layers = [ResidualBlock(num_feats, num_feats) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        outputs = []

        for i in range(self.num_stacks):
            hg = self.hourglasses[i](x)
            feat = self.features[i](hg)
            feat = self.lin_layers[i](feat)
            pred = self.pred_layers[i](feat)
            outputs.append(pred)

            if i < self.num_stacks - 1:
                x = x + self.merge_feats[i](feat) + self.merge_preds[i](pred)

        # return last output (for inference) and all stacks for loss if needed
        return outputs[-1], outputs

    # Optional: model-specific sample visualization hook used by Trainer
    def generate_sample_visualization(self, sample, out_path: str, device: torch.device) -> None:
        """Generate a qualitative visualization for a single sample.

        This method is called by Trainer if present, but is not required
        for using the model. It should not assume any particular dataset
        beyond 'image' being a tensor and 'keypoints' (if present) being
        keypoint coordinates in image space.
        """
        img = sample["image"].unsqueeze(0).to(device)  # (1, C, H, W)
        kpts_gt = sample.get("keypoints")

        with torch.no_grad():
            last, _ = self(img)

        hm = last[0].detach().cpu()  # (K, Hh, Wh)
        K, Hh, Wh = hm.shape
        h_flat = hm.view(K, -1)
        idx = torch.argmax(h_flat, dim=1)
        y = (idx // Wh).float().numpy()
        x = (idx % Wh).float().numpy()
        kpts_pred = np.stack([x, y], axis=1)  # (K, 2)

        # convert normalized image tensor back to [0,1] RGB for plotting
        img_np = sample["image"].detach().cpu().numpy()  # (C, H, W) in normalized space
        # inverse of (img/255 - 0.5) / 0.5  ->  img = (x * 0.5 + 0.5) * 255
        img_np = (img_np * 0.5 + 0.5)  # back to [0,1]
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
        img_np = np.clip(img_np, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)

        # draw GT keypoints if available
        if kpts_gt is not None:
            kpts_gt_np = kpts_gt.detach().cpu().numpy()
            if kpts_gt_np.shape[1] >= 2:
                ax.scatter(kpts_gt_np[:, 0], kpts_gt_np[:, 1], c="lime", s=5, label="gt")

        # scale predicted keypoints from heatmap resolution to image resolution
        H_img, W_img = img_np.shape[0], img_np.shape[1]
        scale_y = H_img / float(Hh)
        scale_x = W_img / float(Wh)
        ax.scatter(kpts_pred[:, 0] * scale_x, kpts_pred[:, 1] * scale_y, c="red", s=5, label="pred")

        ax.axis("off")
        ax.legend(loc="upper right", fontsize=6)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
