# pose/data/heatmap.py

import numpy as np


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_heatmaps(
    keypoints,  # (K, 3) [x,y,vis]
    heatmap_size,  # (H,W)
    image_size,  # (H_img, W_img)
    sigma=1.0
):
    num_kpts = keypoints.shape[0]
    H, W = heatmap_size
    heatmaps = np.zeros((num_kpts, H, W), dtype=np.float32)

    tmp_size = sigma * 3
    feat_stride_x = image_size[1] / W
    feat_stride_y = image_size[0] / H

    for i in range(num_kpts):
        x, y, v = keypoints[i]
        if v < 1:
            continue

        mu_x = int(x / feat_stride_x + 0.5)
        mu_y = int(y / feat_stride_y + 0.5)

        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
            continue

        # Gaussian
        size = 2 * tmp_size + 1
        g = gaussian_2d((int(size), int(size)), sigma=sigma)

        g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], H) - ul[1]

        img_x = max(0, ul[0]), min(br[0], W)
        img_y = max(0, ul[1]), min(br[1], H)

        heatmaps[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps


def generate_heatmaps_3d(
    keypoints3d,  # (K, 4) [x,y,z,vis]
    heatmap_size,  # (H,W)
    image_size,  # (H_img, W_img)
    depth_bins: int = 8,
    depth_range: tuple | None = None,
    sigma_spatial: float = 1.0,
    sigma_depth: float = 1.0,
):
    """Generate volumetric heatmaps per keypoint: (K, D, H, W).

    - keypoints3d: numpy array (K,4) with x,y image coords, z depth, vis
    - depth_range: (z_min, z_max) to normalize depths into bins; if None,
      depths are normalized to [0,1] by min/max of provided nonzero depths.
    """

    num_kpts = keypoints3d.shape[0]
    H, W = heatmap_size
    D = depth_bins

    heatmaps = np.zeros((num_kpts, D, H, W), dtype=np.float32)

    # determine depth normalization
    zs = keypoints3d[:, 2]
    valid_mask = keypoints3d[:, 3] > 0
    if depth_range is None:
        if valid_mask.any():
            zmin = float(zs[valid_mask].min())
            zmax = float(zs[valid_mask].max())
            if zmin == zmax:
                zmin -= 0.5
                zmax += 0.5
        else:
            zmin, zmax = 0.0, 1.0
    else:
        zmin, zmax = depth_range

    feat_stride_x = image_size[1] / W
    feat_stride_y = image_size[0] / H

    tmp_size = sigma_spatial * 3
    size = int(2 * tmp_size + 1)

    for i in range(num_kpts):
        x, y, z, v = keypoints3d[i]
        if v < 1:
            continue

        mu_x = x / feat_stride_x
        mu_y = y / feat_stride_y

        # map z to [0, D-1]
        if zmax - zmin == 0:
            mu_z = (D - 1) / 2.0
        else:
            mu_z = (float(z) - zmin) / (zmax - zmin) * (D - 1)

        # precompute spatial gaussian
        g_spatial = gaussian_2d((size, size), sigma=sigma_spatial)

        mu_x_i = int(mu_x + 0.5)
        mu_y_i = int(mu_y + 0.5)

        ul = [int(mu_x_i - tmp_size), int(mu_y_i - tmp_size)]
        br = [int(mu_x_i + tmp_size + 1), int(mu_y_i + tmp_size + 1)]

        if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
            continue

        g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], H) - ul[1]

        img_x = max(0, ul[0]), min(br[0], W)
        img_y = max(0, ul[1]), min(br[1], H)

        patch = g_spatial[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        for dz in range(D):
            dz_coord = float(dz)
            depth_val = np.exp(-((dz_coord - mu_z) ** 2) / (2 * sigma_depth * sigma_depth))
            heatmaps[i, dz, img_y[0]:img_y[1], img_x[0]:img_x[1]] = patch * depth_val

    # normalize per-keypoint volume to max 1
    for i in range(num_kpts):
        m = heatmaps[i].max()
        if m > 0:
            heatmaps[i] /= m

    return heatmaps
