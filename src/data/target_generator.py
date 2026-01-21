"""
Generate BEV targets with CORRECT nuScenes class mapping
"""
import torch
import numpy as np
import math


class BEVTargetGenerator:
    
    def __init__(
        self,
        x_bound=(-50, 50, 0.5),
        y_bound=(-50, 50, 0.5),
        num_classes=10,
    ):
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.num_classes = num_classes
        
        self.bev_w = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.bev_h = int((y_bound[1] - y_bound[0]) / y_bound[2])
        
        # CORRECTED: nuScenes full class names → simplified names
        self.class_map = {
            'vehicle.car': 'car',
            'vehicle.truck': 'truck',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.trailer': 'trailer',
            'movable_object.barrier': 'barrier',
            'vehicle.motorcycle': 'motorcycle',
            'vehicle.bicycle': 'bicycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.wheelchair': 'pedestrian',
            'human.pedestrian.stroller': 'pedestrian',
            'human.pedestrian.personal_mobility': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'movable_object.pushable_pullable': 'barrier',
            'movable_object.debris': 'barrier',
            'static_object.bicycle_rack': 'barrier',
        }
        
        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
    def boxes_to_bev_targets(self, boxes, class_names):
        """
        Convert 3D boxes to BEV grid targets
        
        Args:
            boxes: List of Box objects (in ego frame)
            class_names: List of nuScenes category names
            
        Returns:
            targets: dict with dense BEV labels
        """
        cls_target = torch.zeros(self.num_classes, self.bev_h, self.bev_w)
        center_target = torch.zeros(2, self.bev_h, self.bev_w)
        dim_target = torch.zeros(2, self.bev_h, self.bev_w)
        rot_target = torch.zeros(2, self.bev_h, self.bev_w)
        
        for box, full_class_name in zip(boxes, class_names):
            # Map to simplified class
            if full_class_name not in self.class_map:
                continue
            
            simple_class = self.class_map[full_class_name]
            
            if simple_class not in self.class_names:
                continue
            
            class_idx = self.class_names.index(simple_class)
            
            # Box center
            center_x = box.center[0]
            center_y = box.center[1]
            
            # Check BEV range
            if not (self.x_bound[0] <= center_x <= self.x_bound[1] and
                    self.y_bound[0] <= center_y <= self.y_bound[1]):
                continue
            
            # Grid coordinates
            grid_x = int((center_x - self.x_bound[0]) / self.x_bound[2])
            grid_y = int((center_y - self.y_bound[0]) / self.y_bound[2])
            
            grid_x = max(0, min(grid_x, self.bev_w - 1))
            grid_y = max(0, min(grid_y, self.bev_h - 1))
            
            # Set targets
            cls_target[class_idx, grid_y, grid_x] = 1.0
            
            # Offset within cell
            offset_x = (center_x - (grid_x * self.x_bound[2] + self.x_bound[0])) / self.x_bound[2]
            offset_y = (center_y - (grid_y * self.y_bound[2] + self.y_bound[0])) / self.y_bound[2]
            center_target[:, grid_y, grid_x] = torch.tensor([offset_x, offset_y])
            
            # Dimensions (width, length from nuScenes wlh)
            width = box.wlh[0]
            length = box.wlh[1]
            dim_target[:, grid_y, grid_x] = torch.tensor([width, length])
            
            # Rotation
            yaw = box.orientation.yaw_pitch_roll[0]
            rot_target[:, grid_y, grid_x] = torch.tensor([math.sin(yaw), math.cos(yaw)])
        
        return {
            'cls': cls_target,
            'center': center_target,
            'dim': dim_target,
            'rot': rot_target,
        }


if __name__ == '__main__':
    print("Testing corrected target generator...")
    
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    
    # Create test boxes with nuScenes class names
    box1 = Box(
        center=[20.0, -15.0, 0.5],
        size=[1.8, 4.5, 1.5],
        orientation=Quaternion(axis=[0, 0, 1], angle=0.3)
    )
    
    box2 = Box(
        center=[10.0, 5.0, 1.0],
        size=[0.7, 0.8, 1.6],
        orientation=Quaternion(axis=[0, 0, 1], angle=0.0)
    )
    
    generator = BEVTargetGenerator()
    
    # Test with full nuScenes class names
    targets = generator.boxes_to_bev_targets(
        [box1, box2],
        ['vehicle.car', 'human.pedestrian.adult']
    )
    
    print(f"\nTargets:")
    print(f"  Classification: {targets['cls'].shape}")
    print(f"  Objects found: {targets['cls'].sum():.0f}")
    print(f"  Classes with objects: {(targets['cls'].sum(dim=(1,2)) > 0).sum():.0f}")
    
    print(f"\n✅ Target generator fixed!")
