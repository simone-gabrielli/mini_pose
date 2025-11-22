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
    sigma=2.0
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
