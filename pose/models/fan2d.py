import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pose.models.base import PoseModel
from pose.registry import register_model


class ConvBlock(nn.Module):
    """Torch7 convBlock clone.

    numIn -> BN -> ReLU -> conv(numIn, numOut/2, 3x3)
      -> concat( identity,
                 BN -> ReLU -> conv(numOut/2, numOut/4, 3x3)
                     -> concat( identity,
                                BN -> ReLU -> conv(numOut/4, numOut/4, 3x3) ) )
    join along channel dim.
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        mid1 = num_out // 2
        mid2 = num_out // 4

        self.bn1 = nn.BatchNorm2d(num_in, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_in, mid1, kernel_size=3, stride=1, padding=1, bias=False
        )

        # branch2 part1
        self.bn2 = nn.BatchNorm2d(mid1, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(
            mid1, mid2, kernel_size=3, stride=1, padding=1, bias=False
        )

        # branch2 part2
        self.bn3 = nn.BatchNorm2d(mid2, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(
            mid2, mid2, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        # main branch
        out1 = self.conv1(self.relu(self.bn1(x)))  # (B, mid1, H, W)

        # second branch
        y = self.conv2(self.relu(self.bn2(out1)))  # (B, mid2, H, W)
        z = self.conv3(self.relu(self.bn3(y)))     # (B, mid2, H, W)

        # concat identity and residual branches: dim=1 (channels)
        out2 = torch.cat([y, z], dim=1)           # (B, mid1, H, W)

        # final concat to reach num_out channels
        out = torch.cat([out1, out2], dim=1)      # (B, num_out, H, W)
        return out


class SkipLayer(nn.Module):
    """Torch7 skipLayer clone.

    If numIn == numOut: Identity.
    Else: BN -> ReLU -> 1x1 conv(numIn -> numOut).
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        if num_in == num_out:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(num_in, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    num_in, num_out, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )

    def forward(self, x):
        return self.proj(x)


class ResidualFan(nn.Module):
    """Torch7 Residual block: [convBlock, skipLayer] + CAddTable."""

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv_block = ConvBlock(num_in, num_out)
        self.skip = SkipLayer(num_in, num_out)

    def forward(self, x):
        return self.conv_block(x) + self.skip(x)


class HourglassFan(nn.Module):
    """Recursive hourglass as in Torch7 `hourglass(n,f,inp)`.

    Args:
        depth: number of down/up levels (n in original code, 4 in FAN2D/3D).
        num_feats: feature channels (nFeats).
        n_modules: number of residual modules at each stage (nModules).
    """

    def __init__(self, depth: int, num_feats: int, n_modules: int):
        super().__init__()
        self.depth = depth
        self.num_feats = num_feats
        self.n_modules = n_modules

        # upper branch residuals at current resolution
        self.up_res = nn.Sequential(
            *[ResidualFan(num_feats, num_feats) for _ in range(n_modules)]
        )

        # lower branch: pool -> residuals
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low_res = nn.Sequential(
            *[ResidualFan(num_feats, num_feats) for _ in range(n_modules)]
        )

        # recursive or bottom part
        if depth > 1:
            self.hg = HourglassFan(depth - 1, num_feats, n_modules)
        else:
            self.hg = nn.Sequential(
                *[ResidualFan(num_feats, num_feats) for _ in range(n_modules)]
            )

        # after recursion: more residuals then upsample
        self.low_res2 = nn.Sequential(
            *[ResidualFan(num_feats, num_feats) for _ in range(n_modules)]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        # Upper branch
        up1 = self.up_res(x)

        # Lower branch
        low1 = self.pool(x)
        low1 = self.low_res(low1)
        low2 = self.hg(low1)
        low3 = self.low_res2(low2)
        up2 = self.upsample(low3)

        return up1 + up2


def lin_layer(num_in: int, num_out: int):
    """Torch7 lin: 1x1 conv + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_out, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )


@register_model("fan_2d")
class Fan2D(PoseModel):
    """PyTorch port of the FAN2D / 3DFAN backbone (ICCV 2017).

    Mirrors the Torch7 implementation used for 2DFAN-4 / 3DFAN-4:
    - Initial stem: 7x7/2 conv -> BN -> ReLU -> Residual(64->128) ->
      MaxPool(2x2) -> Residual(128->128) -> Residual(128->256).
    - 4 stacked hourglass modules with intermediate supervision.
    - Each stack produces K-channel heatmaps.
    """

    def __init__(
        self,
        num_stacks: int = 4,   # FAN-4: 4 stacks
        num_modules: int = 1,  # nModules
        num_feats: int = 256,  # nFeats
        num_keypoints: int = 68,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_modules = num_modules
        self.num_feats = num_feats
        self.num_keypoints = num_keypoints

        # === Stem: match Torch7 FAN ===
        # conv(3,64,7,7,2,2,3,3)
        self.cnv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # r1 = Residual(64,128)
        self.res1 = ResidualFan(64, 128)

        # pool = MaxPooling(2,2,2,2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # r4 = Residual(128,128)
        self.res2 = ResidualFan(128, 128)

        # r5 = Residual(128,nFeats)
        self.res3 = ResidualFan(128, num_feats)

        # === Stacks ===
        self.hourglasses = nn.ModuleList(
            [
                HourglassFan(depth=4, num_feats=num_feats, n_modules=num_modules)
                for _ in range(num_stacks)
            ]
        )
        self.res_layers = nn.ModuleList(
            [
                nn.Sequential(
                    *[ResidualFan(num_feats, num_feats) for _ in range(num_modules)]
                )
                for _ in range(num_stacks)
            ]
        )
        self.lin_layers = nn.ModuleList(
            [lin_layer(num_feats, num_feats) for _ in range(num_stacks)]
        )
        self.pred_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    num_feats,
                    num_keypoints,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in range(num_stacks)
            ]
        )

        # === Fusion between stacks ===
        self.merge_feats = nn.ModuleList(
            [
                nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0)
                for _ in range(num_stacks - 1)
            ]
        )
        self.merge_preds = nn.ModuleList(
            [
                nn.Conv2d(num_keypoints, num_feats, kernel_size=1, stride=1, padding=0)
                for _ in range(num_stacks - 1)
            ]
        )

    def forward(self, x):
        # Stem (256x256 -> 64x64 if input is 256)
        x = self.relu(self.bn1(self.cnv1(x)))  # 128x128
        x = self.res1(x)
        x = self.pool(x)                       # 64x64
        x = self.res2(x)
        x = self.res3(x)

        outputs = []
        inter = x

        for i in range(self.num_stacks):
            hg = self.hourglasses[i](inter)

            ll = self.res_layers[i](hg)
            ll = self.lin_layers[i](ll)

            tmp_out = self.pred_layers[i](ll)
            outputs.append(tmp_out)

            if i < self.num_stacks - 1:
                ll_ = self.merge_feats[i](ll)
                tmp_out_ = self.merge_preds[i](tmp_out)
                inter = inter + ll_ + tmp_out_

        # For training we return both last and all (intermediate supervision).
        # The JIT 3DFAN-4 used in `face_alignment` only returns the final tensor,
        # but this signature is more convenient for your PoseModel.
        return outputs[-1], outputs

    def generate_sample_visualization(
        self, sample, out_path: str, device: torch.device
    ) -> None:
        """Qualitative visualization with predicted keypoints overlay."""
        img = sample["image"].unsqueeze(0).to(device)  # (1, C, H, W)
        kpts_gt = sample.get("keypoints")

        with torch.no_grad():
            last, _ = self(img)
        hm = last[0].detach().cpu()  # (K, Hh, Wh)
        K, Hh, Wh = hm.shape

        # argmax over heatmaps -> (K, 2) in heatmap coords
        h_flat = hm.view(K, -1)
        idx = torch.argmax(h_flat, dim=1)
        y = (idx // Wh).float().numpy()
        x = (idx % Wh).float().numpy()
        kpts_pred = np.stack([x, y], axis=1)

        img_np = sample["image"].detach().cpu().numpy()  # (C, H, W)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)

        # Undo ImageNet normalization used by the dataset pipeline
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            img_np = (img_np * std) + mean
        except Exception:
            pass

        img_np = np.clip(img_np, 0.0, 1.0)

        # Overlay summed heatmaps
        heatmap_sum = hm.sum(axis=0).numpy()
        heatmap_sum = (heatmap_sum - heatmap_sum.min()) / (
            heatmap_sum.max() - heatmap_sum.min() + 1e-8
        )

        import cv2
        heatmap_resized = cv2.resize(
            heatmap_sum,
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        overlay = img_np.copy()
        cmap = plt.get_cmap("jet")
        heatmap_rgb = cmap(heatmap_resized)[:, :, :3]
        alpha = 0.5
        overlay = (1 - alpha) * overlay + alpha * heatmap_rgb

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)

        # draw GT keypoints if available
        if kpts_gt is not None:
            kpts_gt_np = kpts_gt.detach().cpu().numpy()
            if kpts_gt_np.shape[1] >= 2:
                ax.scatter(
                    kpts_gt_np[:, 0], kpts_gt_np[:, 1],
                    c="lime", s=5, label="gt"
                )

        # scale predicted keypoints from heatmap resolution to image resolution
        H_img, W_img = img_np.shape[:2]
        scale_y = H_img / float(Hh)
        scale_x = W_img / float(Wh)
        kpts_pred_img = np.stack(
            [kpts_pred[:, 0] * scale_x, kpts_pred[:, 1] * scale_y],
            axis=1,
        )

        ax.scatter(
            kpts_pred_img[:, 0], kpts_pred_img[:, 1],
            c="red", s=5, label="pred"
        )
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=6)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
