"""
nuScenes Dataset with CORRECTED target generation
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from PIL import Image
import os
from typing import List, Dict, Tuple

from src.data.target_generator import BEVTargetGenerator


class NuScenesMultiViewDataset(Dataset):
    
    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-mini',
        split: str = 'train',
        cameras: List[str] = None,
        image_size: Tuple[int, int] = (224, 400),
        return_targets: bool = True,
    ):
        super().__init__()
        
        self.data_root = data_root
        self.version = version
        self.split = split
        self.image_size = image_size
        self.return_targets = return_targets
        
        if cameras is None:
            self.cameras = [
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
        else:
            self.cameras = cameras
        
        print(f"Loading nuScenes {version} ({split} split)...")
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        
        self.samples = self._create_split()
        print(f"Loaded {len(self.samples)} samples for {split}")
        
        if return_targets:
            self.target_gen = BEVTargetGenerator()
        
    def _create_split(self):
        all_samples = self.nusc.sample
        num_train = int(0.8 * len(all_samples))
        
        if self.split == 'train':
            samples = all_samples[:num_train]
        else:
            samples = all_samples[num_train:]
        
        return [s['token'] for s in samples]
    
    def _get_boxes_and_classes(self, sample_token):
        """
        Get 3D boxes in ego vehicle frame (CORRECTED)
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Get ego pose from lidar
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        boxes = []
        class_names = []
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Skip if visibility is 0 or very low
            if ann['visibility_token'] == '1':  # Not visible
                continue
            
            # Get box - this is already in global frame
            box = Box(
                center=ann['translation'],
                size=ann['size'],
                orientation=Quaternion(ann['rotation'])
            )
            
            # Transform from global to ego
            # Step 1: Translate
            box.translate(-np.array(ego_pose['translation']))
            
            # Step 2: Rotate
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            
            boxes.append(box)
            class_names.append(ann['category_name'])
        
        return boxes, class_names
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Load images
        images = []
        intrinsics = []
        extrinsics = []
        
        for cam_name in self.cameras:
            cam_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            img_path = os.path.join(self.data_root, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]))
            img = np.array(img)
            
            calib = self.nusc.get('calibrated_sensor', 
                                  cam_data['calibrated_sensor_token'])
            
            K = np.array(calib['camera_intrinsic'])
            scale_w = self.image_size[1] / 1600
            scale_h = self.image_size[0] / 900
            K[0, :] *= scale_w
            K[1, :] *= scale_h
            
            rotation = Quaternion(calib['rotation']).rotation_matrix
            translation = np.array(calib['translation'])
            cam_to_ego = np.eye(4)
            cam_to_ego[:3, :3] = rotation
            cam_to_ego[:3, 3] = translation
            
            images.append(img)
            intrinsics.append(K)
            extrinsics.append(cam_to_ego)
        
        images = np.stack(images)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0
        
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()
        extrinsics = torch.from_numpy(np.stack(extrinsics)).float()
        
        result = {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'sample_token': sample_token,
        }
        
        if self.return_targets:
            boxes, class_names = self._get_boxes_and_classes(sample_token)
            targets = self.target_gen.boxes_to_bev_targets(boxes, class_names)
            result['targets'] = targets
        
        return result


if __name__ == '__main__':
    print("Testing corrected dataset...")
    
    dataset = NuScenesMultiViewDataset(
        data_root='data/nuscenes',
        version='v1.0-mini',
        split='train',
        return_targets=True
    )
    
    # Check multiple samples
    print("\nChecking object counts:")
    for idx in range(10):
        sample = dataset[idx]
        num_obj = sample['targets']['cls'].sum().item()
        print(f"  Sample {idx}: {num_obj:.0f} objects")
    
    print(f"\n✅ Testing complete!")
