"""
Inverse Perspective Mapping (IPM)

Assumes a flat ground plane and uses known camera intrinsics/extrinsics to
project each BEV grid point onto the image plane, then samples pixel color
via bilinear interpolation. No learning involved.

nuScenes ego frame: X = forward, Y = left, Z = up
BEV grid convention used here:
    bev_x (lateral) maps to  -Y_ego
    bev_y (forward) maps to  +X_ego
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from src.utils import load_config


# --- single camera ---

def ipm_single_camera(img_tensor, K, cam2ego, cfg):
    """
    Project a BEV grid onto one camera image and sample pixel colors.

    img_tensor : (3, H, W) float32, values in [0, 1]
    K          : (3, 3) float32 - intrinsics scaled to the resized image
    cam2ego    : (4, 4) float32 - camera-to-ego rigid transform
    cfg        : config dict

    Returns
        bev       : (3, bev_h, bev_w) float32
        valid_mask: (bev_h, bev_w) bool - True where the camera covers the cell
    """
    bev_h = cfg['ipm']['bev_h']
    bev_w = cfg['ipm']['bev_w']
    x_min, x_max = cfg['ipm']['x_min'], cfg['ipm']['x_max']
    y_min, y_max = cfg['ipm']['y_min'], cfg['ipm']['y_max']
    img_h, img_w = cfg['data']['img_h'], cfg['data']['img_w']

    # build a flat (Z=0) grid of ego-frame 3D points covering the BEV ROI
    bev_xs = np.linspace(x_min, x_max, bev_w, dtype=np.float32)
    bev_ys = np.linspace(y_max, y_min, bev_h, dtype=np.float32)  # top row = far
    grid_x, grid_y = np.meshgrid(bev_xs, bev_ys)

    X_ego = grid_y.ravel()
    Y_ego = -grid_x.ravel()
    Z_ego = np.zeros_like(X_ego)
    pts_ego = np.stack([X_ego, Y_ego, Z_ego, np.ones_like(X_ego)], axis=0)

    # transform ground points to camera frame
    ego2cam = np.linalg.inv(cam2ego)
    pts_cam = ego2cam @ pts_ego

    # only keep points in front of the camera
    Z_cam = pts_cam[2]
    valid = Z_cam > 0.1

    # perspective projection
    uvw = K @ pts_cam[:3]
    u = uvw[0] / np.clip(uvw[2], 1e-6, None)
    v = uvw[1] / np.clip(uvw[2], 1e-6, None)

    # normalize to [-1, 1] for F.grid_sample
    u_norm = (u / (img_w - 1)) * 2.0 - 1.0
    v_norm = (v / (img_h - 1)) * 2.0 - 1.0

    valid &= (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1)

    grid = np.stack([u_norm, v_norm], axis=-1)
    grid = torch.from_numpy(grid).reshape(1, bev_h, bev_w, 2)

    src = img_tensor.unsqueeze(0)
    bev = F.grid_sample(src, grid, mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True)
    bev = bev.squeeze(0)

    valid_mask = torch.from_numpy(valid.reshape(bev_h, bev_w))
    bev = bev * valid_mask.float().unsqueeze(0)

    return bev, valid_mask


# --- 6-camera stitch ---

def ipm_stitch_all_cameras(imgs, Ks, cam2egos, cfg):
    """
    Run IPM for all 6 cameras and alpha-blend overlapping regions.

    imgs     : (6, 3, H, W)
    Ks       : (6, 3, 3)
    cam2egos : (6, 4, 4)

    Returns  : (3, bev_h, bev_w)
    """
    bev_h = cfg['ipm']['bev_h']
    bev_w = cfg['ipm']['bev_w']

    accum  = torch.zeros(3, bev_h, bev_w)
    weight = torch.zeros(1, bev_h, bev_w)

    for i in range(6):
        bev_i, mask_i = ipm_single_camera(
            imgs[i], Ks[i].numpy(), cam2egos[i].numpy(), cfg)
        w = mask_i.float().unsqueeze(0)
        accum  += bev_i * w
        weight += w

    stitched = accum / (weight + 1e-6)
    stitched = stitched * (weight > 0).float()
    return stitched


# --- visualization ---

def bev_to_bgr(bev_tensor, flip_vertical=True):
    img = bev_tensor.permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if flip_vertical:
        img = cv2.flip(img, 0)
    return img


def draw_ego_marker(bev_bgr, cfg):
    bev_h, bev_w = bev_bgr.shape[:2]
    cx = bev_w // 2
    cy = bev_h - 10
    cv2.circle(bev_bgr, (cx, cy), 8, (255, 255, 255), -1)
    cv2.circle(bev_bgr, (cx, cy), 8, (0, 0, 0), 2)
    return bev_bgr
