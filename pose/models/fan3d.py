import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pose.models.base import PoseModel
from pose.registry import register_model

class ConvBlock(nn.Module):
    # ...existing code...
    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        mid1 = num_out // 2
        mid2 = num_out // 4
        self.bn1 = nn.BatchNorm2d(num_in, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_in, mid1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid1, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(mid1, mid2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid2, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(mid2, mid2, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        out1 = self.conv1(self.relu(self.bn1(x)))
        y = self.conv2(self.relu(self.bn2(out1)))
        z = self.conv3(self.relu(self.bn3(y)))
        out2 = torch.cat([y, z], dim=1)
        out = torch.cat([out1, out2], dim=1)
        return out

class SkipLayer(nn.Module):
    # ...existing code...
    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        if num_in == num_out:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(num_in, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_in, num_out, kernel_size=1, stride=1, padding=0, bias=False),
            )
    def forward(self, x):
        return self.proj(x)

class ResidualFan(nn.Module):
    # ...existing code...
    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv_block = ConvBlock(num_in, num_out)
        self.skip = SkipLayer(num_in, num_out)
    def forward(self, x):
        return self.conv_block(x) + self.skip(x)

class HourglassFan(nn.Module):
    # ...existing code...
    def __init__(self, depth: int, num_feats: int, n_modules: int):
        super().__init__()
        self.depth = depth
        self.num_feats = num_feats
        self.n_modules = n_modules
        self.up_res = nn.Sequential(*[ResidualFan(num_feats, num_feats) for _ in range(n_modules)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low_res = nn.Sequential(*[ResidualFan(num_feats, num_feats) for _ in range(n_modules)])
        if depth > 1:
            self.hg = HourglassFan(depth - 1, num_feats, n_modules)
        else:
            self.hg = nn.Sequential(*[ResidualFan(num_feats, num_feats) for _ in range(n_modules)])
        self.low_res2 = nn.Sequential(*[ResidualFan(num_feats, num_feats) for _ in range(n_modules)])
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, x):
        up1 = self.up_res(x)
        low1 = self.pool(x)
        low1 = self.low_res(low1)
        low2 = self.hg(low1)
        low3 = self.low_res2(low2)
        up2 = self.upsample(low3)
        return up1 + up2

def lin_layer(num_in: int, num_out: int):
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_out, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )

@register_model("fan3d")
class Fan3D(PoseModel):
    """FAN3D: FAN2D with additional depth head."""
    def __init__(
        self,
        num_stacks: int = 8,
        num_modules: int = 1,
        num_feats: int = 256,
        num_keypoints: int = 68,
        in_channels: int = 3,
        depth_channels: int = 1,  # output channels for depth
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_modules = num_modules
        self.num_feats = num_feats
        self.num_keypoints = num_keypoints
        self.depth_channels = depth_channels
        self.cnv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualFan(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualFan(128, 128)
        self.res3 = ResidualFan(128, num_feats)
        self.hourglasses = nn.ModuleList([
            HourglassFan(depth=4, num_feats=num_feats, n_modules=num_modules) for _ in range(num_stacks)
        ])
        self.res_layers = nn.ModuleList([
            nn.Sequential(*[ResidualFan(num_feats, num_feats) for _ in range(num_modules)])
            for _ in range(num_stacks)
        ])
        self.lin_layers = nn.ModuleList([
            lin_layer(num_feats, num_feats) for _ in range(num_stacks)
        ])
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(num_feats, num_keypoints, kernel_size=1, stride=1, padding=0)
            for _ in range(num_stacks)
        ])
        self.depth_layers = nn.ModuleList([
            nn.Conv2d(num_feats, depth_channels, kernel_size=1, stride=1, padding=0)
            for _ in range(num_stacks)
        ])
        self.merge_feats = nn.ModuleList([
            nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0)
            for _ in range(num_stacks - 1)
        ])
        self.merge_preds = nn.ModuleList([
            nn.Conv2d(num_keypoints, num_feats, kernel_size=1, stride=1, padding=0)
            for _ in range(num_stacks - 1)
        ])
    def forward(self, x):
        x = self.relu(self.bn1(self.cnv1(x)))
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
        heatmaps = []
        depths = []
        inter = x
        for i in range(self.num_stacks):
            hg = self.hourglasses[i](inter)
            ll = self.res_layers[i](hg)
            ll = self.lin_layers[i](ll)
            heatmap = self.pred_layers[i](ll)
            depth = self.depth_layers[i](ll)
            heatmaps.append(heatmap)
            depths.append(depth)
            if i < self.num_stacks - 1:
                ll_ = self.merge_feats[i](ll)
                heatmap_ = self.merge_preds[i](heatmap)
                inter = inter + ll_ + heatmap_
        return heatmaps[-1], heatmaps, depths[-1], depths
    # Optional: visualization hook (adapted for depth)
    def generate_sample_visualization(self, sample, out_path: str, device: torch.device) -> None:
        img = sample["image"].unsqueeze(0).to(device)
        kpts_gt = sample.get("keypoints")
        with torch.no_grad():
            last_hm, _, last_depth, _ = self(img)
        hm = last_hm[0].detach().cpu()
        depth_map = last_depth[0].detach().cpu()
        K, Hh, Wh = hm.shape
        h_flat = hm.view(K, -1)
        idx = torch.argmax(h_flat, dim=1)
        y = (idx // Wh).float().numpy()
        x = (idx % Wh).float().numpy()
        kpts_pred = np.stack([x, y], axis=1)
        img_np = sample["image"].detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        if kpts_gt is not None:
            kpts_gt_np = kpts_gt.detach().cpu().numpy()
            if kpts_gt_np.shape[1] >= 2:
                ax.scatter(kpts_gt_np[:, 0], kpts_gt_np[:, 1], c="lime", s=5, label="gt")
        H_img, W_img = img_np.shape[0], img_np.shape[1]
        scale_y = H_img / float(Hh)
        scale_x = W_img / float(Wh)
        kpts_pred_img = np.stack([
            kpts_pred[:, 0] * scale_x, kpts_pred[:, 1] * scale_y
        ], axis=1)
        ax.scatter(kpts_pred_img[:, 0], kpts_pred_img[:, 1], c="red", s=5, label="pred")
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=6)
        fig.tight_layout()
        fig.savefig(out_path.replace(".png", "_kpts.png"), dpi=150)
        plt.close(fig)
        # Depth visualization
        plt.figure(figsize=(4, 4))
        plt.imshow(depth_map[0], cmap="viridis")
        plt.axis("off")
        plt.title("Predicted Depth")
        plt.tight_layout()
        plt.savefig(out_path.replace(".png", "_depth.png"), dpi=150)
        plt.close()
