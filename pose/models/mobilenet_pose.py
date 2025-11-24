import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2

from pose.models.base import PoseModel
from pose.registry import register_model


@register_model("mobilenet_pose")
class MobileNetPose(PoseModel):
    """
    Lightweight heatmap-based pose model using MobileNetV2 backbone.

    - Input:  (B, 3, H, W), e.g. 256x256
    - Output: last_pred: (B, K, H_hm, W_hm), e.g. 64x64 if we use 3 deconvs
             all_preds: [last_pred] to match stacked-hourglass API
    """

    def __init__(
        self,
        num_keypoints: int,
        pretrained: bool = True,
        deconv_channels=(256, 256, 256),
        deconv_kernel=4,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Backbone
        backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = backbone.features  # (B, 1280, H/32, W/32)
        backbone_out_channels = 1280

        # Deconv head: upsample from stride 32 → stride 4 (for 64x64 from 256x256)
        self.deconv_layers = self._make_deconv_layers(
            backbone_out_channels, deconv_channels, deconv_kernel
        )

        self.final_layer = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Init head weights
        self._init_weights()

    def _make_deconv_layers(self, in_channels, num_filters, kernel_size):
        layers = []
        for f in num_filters:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    f,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU(inplace=True))
            in_channels = f
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.final_layer.weight, std=0.001)
        if self.final_layer.bias is not None:
            nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        # Backbone
        feat = self.backbone(x)      # (B, 1280, H/32, W/32) e.g. 8x8 for 256 input
        # Deconv head
        feat = self.deconv_layers(feat)  # (B, C, H_hm, W_hm) e.g. 64x64
        out = self.final_layer(feat)     # (B, K, H_hm, W_hm)

        # To be compatible with your Trainer (expects last_pred, preds_all)
        return out, [out]

    # Optional: model-specific sample visualization hook used by Trainer
    def generate_sample_visualization(self, sample, out_path: str, device: torch.device) -> None:
        """Generate a qualitative visualization for a single sample.

        Decodes the predicted heatmaps to keypoints and overlays them with
        ground-truth keypoints (if available) on the input crop.
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

        # convert tensor back to [0,1] RGB for plotting
        img_np = sample["image"].detach().cpu().numpy()  # (C, H, W)
        # If using [-1,1] normalization, uncomment next line:
        # img_np = (img_np * 0.5 + 0.5)
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
        kpts_pred_img = np.stack(
            [kpts_pred[:, 0] * scale_x, kpts_pred[:, 1] * scale_y],
            axis=1,
        )
        ax.scatter(kpts_pred_img[:, 0], kpts_pred_img[:, 1], c="red", s=5, label="pred")

        ax.axis("off")
        ax.legend(loc="upper right", fontsize=6)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
