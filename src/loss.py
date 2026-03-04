"""
BEV detection losses.

The BEV grid is 200x200 at 0.5m/cell covering [-50, 50]m in X and Y.
The vast majority of cells are empty, so standard cross-entropy would
collapse to predicting all-negative. Focal loss handles this by down-
weighting easy negatives via a modulating factor on the loss term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_config


# --- target builder ---

def build_targets(boxes_list, cfg, device):
    """
    Rasterize 3D bounding boxes onto the BEV grid to produce
    dense supervision targets for all four task heads.

    boxes_list : list of B tensors (N_i, 9)
                 cols: [class_idx, cx, cy, cz, l, w, h, sin_yaw, cos_yaw]
                 cx/cy are in the ego frame (X=forward, Y=left)

    Returns a dict of (B, *, 200, 200) tensors:
        cls    - one-hot class heatmap
        offset - sub-cell center offset
        dim    - box dimensions (l, w)
        rot    - rotation as (sin, cos)
        mask   - 1 at occupied cells, used to gate regression losses
    """
    lss    = cfg['lss']
    x_min, x_max = lss['bev_x_min'], lss['bev_x_max']
    y_min, y_max = lss['bev_y_min'], lss['bev_y_max']
    res    = lss['bev_res']
    bev_w  = int((x_max - x_min) / res)
    bev_h  = int((y_max - y_min) / res)
    num_cls = cfg['classes']['num']
    B = len(boxes_list)

    cls_t = torch.zeros(B, num_cls, bev_h, bev_w, device=device)
    off_t = torch.zeros(B, 2,       bev_h, bev_w, device=device)
    dim_t = torch.zeros(B, 2,       bev_h, bev_w, device=device)
    rot_t = torch.zeros(B, 2,       bev_h, bev_w, device=device)
    mask  = torch.zeros(B, 1,       bev_h, bev_w, device=device)

    for b, boxes in enumerate(boxes_list):
        if boxes.shape[0] == 0:
            continue
        boxes = boxes.to(device)

        cx_ego = boxes[:, 1]  # X_ego (forward)
        cy_ego = boxes[:, 2]  # Y_ego (left)

        # map ego coords to BEV grid indices
        # lateral axis: -Y_ego -> BEV col
        # forward axis:  X_ego -> BEV row
        col_f = (-cy_ego - x_min) / res
        row_f = ( cx_ego - y_min) / res
        col_i = col_f.long()
        row_i = row_f.long()

        valid = (
            (col_i >= 0) & (col_i < bev_w) &
            (row_i >= 0) & (row_i < bev_h)
        )

        for n in range(boxes.shape[0]):
            if not valid[n]:
                continue
            ci, ri = col_i[n].item(), row_i[n].item()
            cls_idx = int(boxes[n, 0].item())

            cls_t[b, cls_idx, ri, ci] = 1.0
            off_t[b, 0,       ri, ci] = col_f[n] - col_i[n].float()
            off_t[b, 1,       ri, ci] = row_f[n] - row_i[n].float()
            dim_t[b, 0,       ri, ci] = boxes[n, 4]
            dim_t[b, 1,       ri, ci] = boxes[n, 5]
            rot_t[b, 0,       ri, ci] = boxes[n, 7]
            rot_t[b, 1,       ri, ci] = boxes[n, 8]
            mask [b, 0,       ri, ci] = 1.0

    return {'cls': cls_t, 'offset': off_t, 'dim': dim_t, 'rot': rot_t, 'mask': mask}


# --- focal loss ---

def focal_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """
    Focal loss following CenterPoint / CornerNet formulation.

    Positive and negative terms are normalized independently:
      - positives  / num_pos   (standard centerness normalization)
      - negatives  / num_elem  (prevents blowup on very sparse grids)
    """
    pred_sig = torch.sigmoid(pred)
    pos_mask = target.eq(1).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = (
        -torch.log(pred_sig.clamp(eps))
        * (1 - pred_sig).pow(alpha)
        * pos_mask
    )
    neg_loss = (
        -torch.log((1 - pred_sig).clamp(eps))
        * pred_sig.pow(alpha)
        * (1 - target).pow(beta)
        * neg_mask
    )

    num_pos  = pos_mask.sum().clamp(min=1)
    num_elem = pred.numel()
    return pos_loss.sum() / num_pos + neg_loss.sum() / num_elem


# --- regression losses ---

def l1_loss_masked(pred, target, mask):
    mask_exp = mask.expand_as(pred)
    num      = mask_exp.sum().clamp(min=1)
    return (pred - target).abs().mul(mask_exp).sum() / num


def smooth_l1_loss_masked(pred, target, mask):
    mask_exp = mask.expand_as(pred)
    num      = mask_exp.sum().clamp(min=1)
    return F.smooth_l1_loss(pred, target, reduction='none').mul(mask_exp).sum() / num


# --- combined loss ---

class BEVLoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.w_cls = 1.0
        self.w_off = 1.0
        self.w_dim = 0.5
        self.w_rot = 0.5

    def forward(self, preds, targets):
        mask = targets['mask']

        loss_cls = focal_loss(preds['cls'], targets['cls'])
        loss_off = l1_loss_masked(preds['offset'], targets['offset'], mask)

        # regress log-dimensions to keep gradients well-scaled
        loss_dim = l1_loss_masked(
            torch.log(preds['dim'].clamp(1e-3)),
            torch.log(targets['dim'].clamp(1e-3)),
            mask,
        )
        loss_rot = smooth_l1_loss_masked(preds['rot'], targets['rot'], mask)

        total = (self.w_cls * loss_cls + self.w_off * loss_off +
                 self.w_dim * loss_dim + self.w_rot * loss_rot)

        return {'total': total, 'cls': loss_cls,
                'off': loss_off, 'dim': loss_dim, 'rot': loss_rot}
