"""
Debug: Find out where objects actually are
"""
import sys
sys.path.append('.')

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)

print("="*70)
print("DEBUGGING OBJECT LOCATIONS")
print("="*70)

# Check first 5 samples
for sample_idx in range(5):
    sample = nusc.sample[sample_idx]
    
    print(f"\n{'='*70}")
    print(f"SAMPLE {sample_idx}")
    print(f"{'='*70}")
    print(f"Total annotations: {len(sample['anns'])}")
    
    # Get ego pose
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    print(f"\nEgo position (global): {ego_pose['translation']}")
    
    # Check each annotation
    for i, ann_token in enumerate(sample['anns'][:5]):  # First 5 objects
        ann = nusc.get('sample_annotation', ann_token)
        
        # Global position
        global_pos = np.array(ann['translation'])
        
        # Transform to ego frame
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation'])
        )
        
        # Global to ego
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        ego_pos = box.center
        
        print(f"\nObject {i}: {ann['category_name']}")
        print(f"  Global position: [{global_pos[0]:.1f}, {global_pos[1]:.1f}, {global_pos[2]:.1f}]")
        print(f"  Ego position: [{ego_pos[0]:.1f}, {ego_pos[1]:.1f}, {ego_pos[2]:.1f}]")
        print(f"  Distance from ego: {np.linalg.norm(ego_pos[:2]):.1f}m")
        print(f"  Size (W×L×H): {ann['size']}")
        print(f"  Visibility: {ann['visibility_token']}")
        
        # Check if in our BEV range
        in_x = -50 <= ego_pos[0] <= 50
        in_y = -50 <= ego_pos[1] <= 50
        in_range = in_x and in_y
        print(f"  In BEV range (-50 to 50m)? {in_range}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
