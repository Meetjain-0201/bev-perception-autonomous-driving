"""
Generate the side-by-side BEV demo video.

Layout: [Front Camera | Classical IPM BEV | Ground Truth BEV]

Run from the project root:
    python scripts/demo_video.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch

from src.dataset import get_dataset, CAMERAS
from src.ipm     import ipm_stitch_all_cameras, bev_to_bgr, draw_ego_marker
from src.utils   import load_config, ensure_dir

CAM_W    = 640
CAM_H    = 360
PANEL_W  = 500
PANEL_H  = 500
TITLE_H  = 44
LEGEND_H = 32
FPS      = 5

BEV_RANGE = 40.0  # metres shown around the ego vehicle in the GT panel

CLASS_NAMES = [
    'car', 'truck', 'constr.veh', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'cone',
]
CLASS_COLORS = [
    (0,   80,  255),  # car
    (0,   165, 255),  # truck
    (0,   255, 255),  # construction vehicle
    (255, 50,  50 ),  # bus
    (200, 0,   180),  # trailer
    (140, 140, 140),  # barrier
    (0,   220, 100),  # motorcycle
    (255, 230, 0  ),  # bicycle
    (255, 160, 80 ),  # pedestrian
    (180, 0,   255),  # traffic cone
]


def make_title_bar(total_w, frame_idx, n_frames):
    bar = np.zeros((TITLE_H, total_w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    txt = (f'BEV Perception Demo  |  Frame {frame_idx+1}/{n_frames}'
           f'  |  Classical IPM  vs  Ground Truth BEV')
    cv2.putText(bar, txt, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def make_legend_strip(total_w):
    strip = np.zeros((LEGEND_H, total_w, 3), dtype=np.uint8)
    strip[:] = (18, 18, 18)
    x = 10
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        cv2.rectangle(strip, (x, 9), (x + 14, 23), color, -1)
        cv2.putText(strip, name, (x + 18, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
        x += len(name) * 7 + 32
        if x > total_w - 100:
            break
    return strip


def make_panel_label(img, label, color=(255, 255, 255)):
    cv2.putText(img, label, (9, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (9, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,     1, cv2.LINE_AA)
    return img


def draw_rotated_box(canvas, cx_px, cy_px, l_px, w_px, yaw_rad, color, thickness=2):
    """Draw an oriented bounding box. l_px is along the heading, w_px is lateral."""
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    hl, hw = l_px / 2.0, w_px / 2.0

    corners = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]])
    rot     = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners = (rot @ corners.T).T  # (4, 2) in ego-offset space

    pts = np.zeros((4, 2), dtype=np.int32)
    for i, (fwd, lat) in enumerate(corners):
        pts[i, 0] = int(cx_px + lat)
        pts[i, 1] = int(cy_px - fwd)

    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

    # short line indicating heading direction
    front_x = int(cx_px + hl * (-sin_y))
    front_y = int(cy_px - hl *  cos_y)
    cv2.line(canvas, (int(cx_px), int(cy_px)), (front_x, front_y),
             color, thickness + 1)


def gt_bev_panel(boxes, panel_h, panel_w, bev_range=BEV_RANGE):
    """
    Render a clean top-down map: ego vehicle at centre, annotated objects
    drawn as oriented boxes, concentric range rings every 10 m.

    boxes: list of dicts (class_idx, cx, cy, l, w, yaw) in ego frame
           nuScenes convention: X = forward, Y = left
    """
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 18)

    cx0   = panel_w // 2
    cy0   = panel_h // 2
    scale = (panel_w / 2) / bev_range

    for d in range(10, int(bev_range) + 1, 10):
        cv2.circle(canvas, (cx0, cy0), int(d * scale), (35, 35, 45), 1, cv2.LINE_AA)

    cv2.line(canvas, (cx0, 0),       (cx0, panel_h),  (40, 40, 55), 1)
    cv2.line(canvas, (0,   cy0),     (panel_w, cy0),  (40, 40, 55), 1)

    for b in boxes:
        fwd_m = b['cx']
        lat_m = b['cy']
        if abs(fwd_m) > bev_range or abs(lat_m) > bev_range:
            continue

        px    = int(cx0 + lat_m * scale)
        py    = int(cy0 - fwd_m * scale)
        color = CLASS_COLORS[b['class_idx']]

        # nuScenes ann['size'] = [width, length, height], stored as l=width, w=length
        # swap so that l_px is drawn along the heading axis
        l_px = max(4, b['w'] * scale)
        w_px = max(3, b['l'] * scale)

        draw_rotated_box(canvas, px, py, l_px, w_px, b['yaw'], color)
        cv2.circle(canvas, (px, py), 3, color, -1)

    # ego vehicle box
    ego_l = int(4.5 * scale)
    ego_w = int(2.0 * scale)
    ego_pts = np.array([
        [cx0 - ego_w // 2, cy0 - ego_l // 2],
        [cx0 + ego_w // 2, cy0 - ego_l // 2],
        [cx0 + ego_w // 2, cy0 + ego_l // 2],
        [cx0 - ego_w // 2, cy0 + ego_l // 2],
    ], dtype=np.int32)
    cv2.fillPoly(canvas,   [ego_pts], (0, 200, 80))
    cv2.polylines(canvas,  [ego_pts], True, (255, 255, 255), 1)
    cv2.line(canvas,
             (cx0 - ego_w // 2, cy0 - ego_l // 4),
             (cx0 + ego_w // 2, cy0 - ego_l // 4),
             (200, 255, 200), 1)

    # flip here so lateral direction matches the camera view
    canvas = cv2.flip(canvas, 1)

    # draw text after the flip so it isn't mirrored
    cv2.putText(canvas, 'FWD', (cx0 - 16, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 160, 80), 1)
    for d in range(10, int(bev_range) + 1, 10):
        label_y = cy0 - int(d * scale) - 3
        if label_y > 10:
            cv2.putText(canvas, f'{d}m', (cx0 + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 90), 1)

    return canvas


def ipm_panel(imgs, Ks, exts, cfg):
    stitched = ipm_stitch_all_cameras(imgs, Ks, exts, cfg)
    bgr      = bev_to_bgr(stitched, flip_vertical=True)
    bgr      = draw_ego_marker(bgr, cfg)
    bgr      = cv2.flip(bgr, 1)
    bgr      = cv2.rotate(bgr, cv2.ROTATE_180)
    bgr      = cv2.resize(bgr, (PANEL_W, PANEL_H))
    return bgr


def front_cam_panel(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (CAM_W, CAM_H))


def main():
    cfg = load_config()
    ensure_dir('results/videos')

    print('Loading dataset...')
    all_items = list(get_dataset('train', cfg)) + list(get_dataset('val', cfg))
    n_frames  = len(all_items)
    print(f'Total frames: {n_frames}')

    total_w  = CAM_W + PANEL_W * 2
    total_h  = TITLE_H + PANEL_H + LEGEND_H
    out_path = 'results/videos/bev_demo.mp4'
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               FPS, (total_w, total_h))

    legend = make_legend_strip(total_w)
    print(f'Output: {total_w}x{total_h} @ {FPS} fps')
    print(f'\nRendering {n_frames} frames...')

    for i, item in enumerate(all_items):
        imgs         = item['images']
        Ks           = item['intrinsics']
        exts         = item['extrinsics']
        boxes_tensor = item['boxes']

        boxes = [{
            'class_idx': int(row[0].item()),
            'cx':  row[1].item(),
            'cy':  row[2].item(),
            'cz':  row[3].item(),
            'l':   row[4].item(),
            'w':   row[5].item(),
            'h':   row[6].item(),
            'yaw': float(torch.atan2(row[7], row[8]).item()),
        } for row in boxes_tensor]

        cam_bgr = front_cam_panel(imgs[0])
        make_panel_label(cam_bgr, 'Front Camera')

        ipm_bgr = ipm_panel(imgs, Ks, exts, cfg)
        make_panel_label(ipm_bgr, 'Classical IPM BEV', color=(255, 220, 100))

        gt_bgr = gt_bev_panel(boxes, PANEL_H, PANEL_W)
        make_panel_label(gt_bgr, 'Ground Truth BEV', color=(100, 255, 150))

        pad_h = PANEL_H - CAM_H
        if pad_h > 0:
            cam_bgr = np.vstack([cam_bgr, np.zeros((pad_h, CAM_W, 3), dtype=np.uint8)])

        frame = np.vstack([
            make_title_bar(total_w, i, n_frames),
            np.hstack([cam_bgr, ipm_bgr, gt_bgr]),
            legend,
        ])
        writer.write(frame)

        if (i + 1) % 40 == 0 or i == 0:
            print(f'  Frame {i+1}/{n_frames}  |  objects: {len(boxes)}')

    writer.release()
    print(f'\nSaved -> {out_path}')

    linkedin_path = 'results/videos/bev_demo_linkedin.mp4'
    ret = os.system(
        f'ffmpeg -y -i {out_path} -vcodec libx264 -pix_fmt yuv420p '
        f'-crf 20 {linkedin_path} 2>/dev/null'
    )
    if ret == 0:
        print(f'Web-compatible copy -> {linkedin_path}')


if __name__ == '__main__':
    main()
