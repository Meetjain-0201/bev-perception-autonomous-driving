"""
Lift-Splat-Shoot (LSS) BEV Perception Model
Philion & Fidler, ECCV 2020 — https://arxiv.org/abs/2008.05711

Each camera image is independently encoded into a feature volume that spans
a set of discrete depth bins (Lift). All six camera frustums are then pooled
into a shared BEV voxel grid (Splat). A BEV CNN processes the resulting
top-down feature map for object detection.

Input:  (B, 6, 3, H, W) multi-camera images
Output: per-cell predictions on a 200x200 BEV grid at 0.5m/cell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from src.utils import load_config


# --- image encoder ---

class ImageEncoder(nn.Module):
    """Per-image feature extraction. Images are processed independently
    before the view transform merges them in 3D space."""

    def __init__(self, neck_out=512):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=True,
            features_only=True, out_indices=[3]
        )
        self.neck = nn.Sequential(
            nn.Conv2d(1024, neck_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.neck(self.backbone(x)[0])


# --- depth net ---

class DepthNet(nn.Module):
    """Predicts a categorical depth distribution over D bins for each
    feature map pixel. Softmax ensures the weights sum to 1 per pixel,
    so the Lift step is an attention-weighted average over depths."""

    def __init__(self, in_ch, d_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, d_bins, 1),
        )

    def forward(self, x):
        return self.net(x).softmax(dim=1)


# --- view transformer ---

class ViewTransformer(nn.Module):
    """
    Lift-Splat view transformation.

    Lift: multiply each feature map pixel by its predicted depth weights,
          producing a weighted feature for each (pixel, depth) frustum point.

    Splat: unproject each frustum point into ego-frame 3D space using
           camera intrinsics and extrinsics, then bin into a BEV voxel grid
           using scatter_add_. The result is averaged across overlapping points.

    All operations are vectorized except the outer loop over (batch, camera),
    which is kept for memory safety on 8GB VRAM.
    """

    def __init__(self, cfg):
        super().__init__()
        lss = cfg['lss']
        self.d_min, self.d_max, self.d_steps = lss['d_min'], lss['d_max'], lss['d_steps']
        self.x_min, self.x_max = lss['bev_x_min'], lss['bev_x_max']
        self.y_min, self.y_max = lss['bev_y_min'], lss['bev_y_max']
        self.z_min, self.z_max = lss['bev_z_min'], lss['bev_z_max']
        self.res    = lss['bev_res']
        self.z_bins = lss['bev_z_bins']
        self.bev_w  = int((self.x_max - self.x_min) / self.res)
        self.bev_h  = int((self.y_max - self.y_min) / self.res)

        # register_buffer ensures this tensor moves to GPU with model.to(device)
        self.register_buffer('frustum', torch.linspace(self.d_min, self.d_max, self.d_steps))

    def forward(self, feats, depth_probs, Ks, cam2egos):
        B6, C, fH, fW = feats.shape
        B   = Ks.shape[0]
        dev = feats.device

        # Lift: depth-weight the features at each pixel
        # result shape: (B6, fH*fW*D, C)
        d_w      = depth_probs.permute(0, 2, 3, 1).unsqueeze(-1)  # (B6,fH,fW,D,1)
        f_exp    = feats.permute(0, 2, 3, 1).unsqueeze(3)          # (B6,fH,fW,1,C)
        weighted = (d_w * f_exp).reshape(B6, fH * fW * self.d_steps, C)

        us = torch.linspace(0, fW - 1, fW, device=dev)
        vs = torch.linspace(0, fH - 1, fH, device=dev)
        grid_v, grid_u = torch.meshgrid(vs, us, indexing='ij')

        depths = self.frustum

        voxel_out = torch.zeros(B, C, self.bev_h, self.bev_w, device=dev, dtype=feats.dtype)
        count_out = torch.zeros(B, 1, self.bev_h, self.bev_w, device=dev, dtype=feats.dtype)

        for b in range(B):
            for cam_i in range(6):
                K      = Ks[b, cam_i]
                c2e    = cam2egos[b, cam_i]
                feat_bc = weighted[b * 6 + cam_i]

                # convert feature-map pixel coords to full-resolution equivalents
                # so the stored intrinsics (calibrated at full res) apply correctly
                img_w_approx = K[0, 2].item() * 2
                img_h_approx = K[1, 2].item() * 2
                u_full = grid_u * (img_w_approx / fW)
                v_full = grid_v * (img_h_approx / fH)

                ones   = torch.ones_like(u_full)
                px     = torch.stack([u_full, v_full, ones], dim=-1)  # (fH,fW,3)
                K_inv  = torch.inverse(K.float())
                dirs   = K_inv @ px.reshape(-1, 3).T                  # (3, fH*fW)

                # scale rays by each depth bin -> (3, fH*fW*D)
                pts_cam = (dirs.unsqueeze(-1) * depths.unsqueeze(0).unsqueeze(0)
                           ).reshape(3, -1)

                ones4   = torch.ones(1, pts_cam.shape[1], device=dev, dtype=pts_cam.dtype)
                pts_ego = (c2e.float() @ torch.cat([pts_cam, ones4], dim=0))[:3]

                # Splat: bin ego-frame points into the BEV voxel grid
                xi = ((pts_ego[0] - self.x_min) / self.res).long()
                yi = ((pts_ego[1] - self.y_min) / self.res).long()
                zi = ((pts_ego[2] - self.z_min) /
                      ((self.z_max - self.z_min) / self.z_bins)).long()

                valid = (
                    (xi >= 0) & (xi < self.bev_w) &
                    (yi >= 0) & (yi < self.bev_h) &
                    (zi >= 0) & (zi < self.z_bins)
                )

                flat_idx = yi[valid] * self.bev_w + xi[valid]
                feat_v   = feat_bc[valid]

                voxel_flat = torch.zeros(self.bev_h * self.bev_w, C,
                                         device=dev, dtype=feats.dtype)
                voxel_flat.scatter_add_(
                    0,
                    flat_idx.unsqueeze(1).expand(-1, C),
                    feat_v.to(voxel_flat.dtype),
                )
                voxel_out[b] += voxel_flat.reshape(self.bev_h, self.bev_w, C).permute(2, 0, 1)

                cnt_flat = torch.zeros(self.bev_h * self.bev_w, 1,
                                        device=dev, dtype=feats.dtype)
                cnt_flat.scatter_add_(
                    0, flat_idx.unsqueeze(1),
                    torch.ones(flat_idx.shape[0], 1, device=dev, dtype=feats.dtype),
                )
                count_out[b] += cnt_flat.reshape(self.bev_h, self.bev_w, 1).permute(2, 0, 1)

        return voxel_out / (count_out + 1e-6)


# --- bev encoder ---

class BEVEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# --- detection head ---

class DetectionHead(nn.Module):
    """Four parallel 1x1 conv heads predicting class heatmap, center offset,
    box dimensions, and rotation (sin/cos encoding)."""

    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.cls    = nn.Conv2d(in_ch, num_classes, 1)
        self.offset = nn.Conv2d(in_ch, 2, 1)
        self.dim    = nn.Conv2d(in_ch, 2, 1)
        self.rot    = nn.Conv2d(in_ch, 2, 1)

    def forward(self, x):
        x = self.shared(x)
        return {
            'cls':    self.cls(x),
            'offset': self.offset(x),
            'dim':    torch.exp(self.dim(x)),  # exp keeps dimensions positive
            'rot':    self.rot(x),
        }


# --- full model ---

class LSSModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        neck_out = cfg['lss']['neck_out']
        bev_ch   = cfg['lss']['bev_channels']
        d_bins   = cfg['lss']['d_steps']
        num_cls  = cfg['classes']['num']

        self.encoder     = ImageEncoder(neck_out)
        self.depth_net   = DepthNet(neck_out, d_bins)
        self.view_tf     = ViewTransformer(cfg)
        self.bev_encoder = BEVEncoder(neck_out, bev_ch)
        self.det_head    = DetectionHead(bev_ch, num_cls)

    def forward(self, images, intrinsics, extrinsics):
        """
        images     : (B, 6, 3, H, W)
        intrinsics : (B, 6, 3, 3)
        extrinsics : (B, 6, 4, 4)  cam-to-ego

        Returns a dict of (B, *, 200, 200) BEV prediction maps.
        bev_features is also returned for downstream visualization.
        """
        B, N, C, H, W = images.shape
        imgs_flat   = images.reshape(B * N, C, H, W)
        feats       = self.encoder(imgs_flat)
        depth_probs = self.depth_net(feats)
        bev         = self.view_tf(feats, depth_probs, intrinsics, extrinsics)
        bev_enc     = self.bev_encoder(bev)
        out         = self.det_head(bev_enc)
        out['bev_features'] = bev_enc
        return out
