"""
nuScenes Multi-View Dataset Loader for BEV Perception
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
import os
from typing import List, Dict, Tuple  # Added Tuple here


class NuScenesMultiViewDataset(Dataset):
    """
    Load multi-view camera images from nuScenes
    
    Returns synchronized images from 6 cameras with calibration data
    """
    
    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-mini',
        split: str = 'train',
        cameras: List[str] = None,
        image_size: Tuple[int, int] = (224, 400),
    ):
        """
        Args:
            data_root: Path to nuScenes dataset
            version: Dataset version (v1.0-mini, v1.0-trainval, etc.)
            split: 'train' or 'val'
            cameras: List of camera names to use
            image_size: (H, W) to resize images
        """
        super().__init__()
        
        self.data_root = data_root
        self.version = version
        self.split = split
        self.image_size = image_size
        
        # Default 6-camera setup
        if cameras is None:
            self.cameras = [
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
        else:
            self.cameras = cameras
        
        # Load nuScenes
        print(f"Loading nuScenes {version} ({split} split)...")
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        
        # Get sample tokens for this split
        self.samples = self._create_split()
        print(f"Loaded {len(self.samples)} samples for {split}")
        
    def _create_split(self):
        """Simple train/val split (80/20)"""
        all_samples = self.nusc.sample
        
        # Use first 80% for train, last 20% for val
        num_train = int(0.8 * len(all_samples))
        
        if self.split == 'train':
            samples = all_samples[:num_train]
        else:  # val
            samples = all_samples[num_train:]
        
        return [s['token'] for s in samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get multi-view images and camera calibration for one sample
        
        Returns:
            dict with:
                - images: (N_cam, 3, H, W) tensor
                - intrinsics: (N_cam, 3, 3) camera matrices
                - extrinsics: (N_cam, 4, 4) cam-to-ego transforms
                - sample_token: str
        """
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        images = []
        intrinsics = []
        extrinsics = []
        
        for cam_name in self.cameras:
            # Get camera data
            cam_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            # Load and resize image
            img_path = os.path.join(self.data_root, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]))  # (W, H)
            img = np.array(img)
            
            # Get camera calibration
            calib = self.nusc.get('calibrated_sensor', 
                                  cam_data['calibrated_sensor_token'])
            
            # Intrinsics (3x3 matrix)
            K = np.array(calib['camera_intrinsic'])
            
            # Adjust intrinsics for resized image
            # Original nuScenes images are 1600x900
            scale_w = self.image_size[1] / 1600
            scale_h = self.image_size[0] / 900
            K[0, :] *= scale_w  # fx, cx
            K[1, :] *= scale_h  # fy, cy
            
            # Extrinsics: camera to ego vehicle (4x4 matrix)
            rotation = Quaternion(calib['rotation']).rotation_matrix
            translation = np.array(calib['translation'])
            
            cam_to_ego = np.eye(4)
            cam_to_ego[:3, :3] = rotation
            cam_to_ego[:3, 3] = translation
            
            images.append(img)
            intrinsics.append(K)
            extrinsics.append(cam_to_ego)
        
        # Convert to tensors
        images = np.stack(images)  # (N_cam, H, W, 3)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)  # (N_cam, 3, H, W)
        images = images / 255.0  # Normalize to [0, 1]
        
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()
        extrinsics = torch.from_numpy(np.stack(extrinsics)).float()
        
        return {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'sample_token': sample_token,
        }


if __name__ == '__main__':
    # Test dataset
    print("Testing dataset loader...")
    
    dataset = NuScenesMultiViewDataset(
        data_root='data/nuscenes',
        version='v1.0-mini',
        split='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Images: {sample['images'].shape}")
    print(f"  Intrinsics: {sample['intrinsics'].shape}")
    print(f"  Extrinsics: {sample['extrinsics'].shape}")
    print(f"  Token: {sample['sample_token']}")
    
    print("\n✅ Dataset loader working!")
