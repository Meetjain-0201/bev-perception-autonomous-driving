import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from src.utils import load_config

# nuScenes provides 6 cameras in this order around the vehicle
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',   'CAM_FRONT_LEFT',
]


def get_nusc(cfg):
    return NuScenes(version=cfg['data']['version'],
                    dataroot=cfg['data']['dataroot'],
                    verbose=False)


# --- helpers ---

def quat_to_mat(q, t):
    rot = Quaternion(q).rotation_matrix
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3,  3] = t
    return mat


def get_cam_calibration(nusc, sample, cam_name, img_h, img_w):
    sd_token = sample['data'][cam_name]
    sd       = nusc.get('sample_data', sd_token)
    cs       = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    K = np.array(cs['camera_intrinsic'], dtype=np.float32)

    img_path = os.path.join(nusc.dataroot, sd['filename'])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_w, img_h))

    # scale focal length and principal point to match the resized resolution
    K_scaled = K.copy()
    K_scaled[0, :] *= img_w / 1600.0
    K_scaled[1, :] *= img_h / 900.0

    cam2ego = quat_to_mat(cs['rotation'], cs['translation'])

    return img, K_scaled, cam2ego


def get_boxes_in_ego(nusc, sample, class_mapping):
    """
    Returns all annotated 3D boxes transformed into the ego vehicle frame
    for a given sample. Yaw is recovered via quaternion composition to avoid
    numerical issues with matrix orthogonality.
    """
    ego_token  = sample['data']['CAM_FRONT']
    sd         = nusc.get('sample_data', ego_token)
    ep         = nusc.get('ego_pose', sd['ego_pose_token'])
    ego2global = quat_to_mat(ep['rotation'], ep['translation'])
    global2ego = np.linalg.inv(ego2global)

    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]
    name2idx = {n: i for i, n in enumerate(class_names)}

    boxes = []
    for ann_token in sample['anns']:
        ann      = nusc.get('sample_annotation', ann_token)
        cat      = ann['category_name']
        if cat not in class_mapping:
            continue
        cls_name = class_mapping[cat]
        if cls_name not in name2idx:
            continue

        center_g = np.array(ann['translation'] + [1.0], dtype=np.float32)
        center_e = (global2ego @ center_g)[:3]

        # compose quaternions directly rather than extracting the rotation
        # submatrix from global2ego, which loses orthogonality at float32
        rot_g   = Quaternion(ann['rotation'])
        ego_q   = Quaternion(ep['rotation'])
        rot_ego = ego_q.inverse * rot_g
        yaw     = rot_ego.yaw_pitch_roll[0]

        l, w, h = ann['size']
        boxes.append({
            'class_idx': name2idx[cls_name],
            'cx': center_e[0], 'cy': center_e[1], 'cz': center_e[2],
            'l': l, 'w': w, 'h': h, 'yaw': yaw,
        })
    return boxes


def boxes_to_tensor(boxes):
    """Pack box dicts into a float32 tensor of shape (N, 9).
    Columns: [class_idx, cx, cy, cz, l, w, h, sin(yaw), cos(yaw)]
    """
    if not boxes:
        return torch.zeros((0, 9), dtype=torch.float32)
    rows = []
    for b in boxes:
        rows.append([
            b['class_idx'],
            b['cx'], b['cy'], b['cz'],
            b['l'],  b['w'],  b['h'],
            np.sin(b['yaw']), np.cos(b['yaw']),
        ])
    return torch.tensor(rows, dtype=torch.float32)


class NuScenesDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg   = cfg
        self.img_h = cfg['data']['img_h']
        self.img_w = cfg['data']['img_w']
        self.nusc  = get_nusc(cfg)
        self.cls_map = cfg['classes']['mapping']

        # mini has 10 scenes; reserve last 2 for validation
        all_scenes = self.nusc.scene
        scenes = all_scenes[:8] if split == 'train' else all_scenes[8:]

        self.samples = []
        for scene in scenes:
            token = scene['first_sample_token']
            while token:
                sample = self.nusc.get('sample', token)
                self.samples.append(sample)
                token = sample['next']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        imgs, Ks, exts = [], [], []
        for cam in CAMERAS:
            img, K, cam2ego = get_cam_calibration(
                self.nusc, sample, cam, self.img_h, self.img_w)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            imgs.append(img)
            Ks.append(torch.from_numpy(K))
            exts.append(torch.from_numpy(cam2ego))

        imgs = torch.stack(imgs)   # (6, 3, H, W)
        Ks   = torch.stack(Ks)    # (6, 3, 3)
        exts = torch.stack(exts)  # (6, 4, 4)

        raw_boxes = get_boxes_in_ego(self.nusc, sample, self.cls_map)
        boxes     = boxes_to_tensor(raw_boxes)

        return {
            'images':     imgs,
            'intrinsics': Ks,
            'extrinsics': exts,
            'boxes':      boxes,
            'token':      sample['token'],
        }


# --- factories ---

def get_dataset(split='train', cfg=None):
    if cfg is None:
        cfg = load_config()
    return NuScenesDataset(cfg, split=split)


def collate_fn(batch):
    # boxes are variable length per sample, so keep them as a list
    return {
        'images':     torch.stack([b['images']     for b in batch]),
        'intrinsics': torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics': torch.stack([b['extrinsics'] for b in batch]),
        'boxes':      [b['boxes'] for b in batch],
        'token':      [b['token'] for b in batch],
    }
