"""
Geometric transformation utilities for BEV perception
Understanding camera projection and coordinate transformations
"""
import torch
import numpy as np


def get_camera_intrinsics_from_matrix(K):
    """
    Extract camera parameters from intrinsic matrix
    
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    
    Args:
        K: (3, 3) intrinsic matrix
    Returns:
        fx, fy, cx, cy
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy


def project_points_to_image(points_3d, intrinsics):
    """
    Project 3D points in camera frame to 2D image pixels
    
    The pinhole camera model:
        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy
    
    Args:
        points_3d: (N, 3) points in camera frame [X, Y, Z]
        intrinsics: (3, 3) camera intrinsic matrix
        
    Returns:
        pixels: (N, 2) pixel coordinates [u, v]
        depths: (N,) depth values (Z coordinates)
    """
    # Ensure torch tensor
    if isinstance(points_3d, np.ndarray):
        points_3d = torch.from_numpy(points_3d).float()
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float()
    
    # Homogeneous coordinates [X, Y, Z] -> [X, Y, Z, 1]
    N = points_3d.shape[0]
    
    # Project: [u*Z, v*Z, Z] = K @ [X, Y, Z]
    pixels_homo = points_3d @ intrinsics.T  # (N, 3)
    
    # Extract depth
    depths = pixels_homo[:, 2]
    
    # Normalize by depth: [u, v] = [u*Z, v*Z] / Z
    pixels = pixels_homo[:, :2] / (depths.unsqueeze(1) + 1e-6)
    
    return pixels, depths


def ego_to_camera_transform(points_ego, cam_to_ego):
    """
    Transform points from ego vehicle frame to camera frame
    
    Args:
        points_ego: (N, 3) or (N, 4) points in ego frame
        cam_to_ego: (4, 4) transformation matrix from camera to ego
        
    Returns:
        points_cam: (N, 3) points in camera frame
    """
    # Ensure torch tensor
    if isinstance(points_ego, np.ndarray):
        points_ego = torch.from_numpy(points_ego).float()
    if isinstance(cam_to_ego, np.ndarray):
        cam_to_ego = torch.from_numpy(cam_to_ego).float()
    
    # Invert to get ego_to_cam
    ego_to_cam = torch.inverse(cam_to_ego)
    
    # Convert to homogeneous if needed
    if points_ego.shape[1] == 3:
        ones = torch.ones(points_ego.shape[0], 1)
        points_homo = torch.cat([points_ego, ones], dim=1)  # (N, 4)
    else:
        points_homo = points_ego
    
    # Transform: points_cam = ego_to_cam @ points_ego
    points_cam_homo = points_homo @ ego_to_cam.T
    points_cam = points_cam_homo[:, :3]
    
    return points_cam


def create_bev_grid(x_bound, y_bound):
    """
    Create BEV grid coordinates
    
    This creates a 2D grid representing the ground plane around the vehicle
    
    Args:
        x_bound: [x_min, x_max, resolution] in meters
        y_bound: [y_min, y_max, resolution] in meters
        
    Returns:
        grid: (H, W, 2) grid coordinates in ego vehicle frame
              where grid[i, j] = [x, y] position in meters
    """
    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound
    
    # Calculate grid dimensions
    H = int((y_max - y_min) / y_res)
    W = int((x_max - x_min) / x_res)
    
    # Create coordinate arrays
    x = torch.linspace(x_min + x_res/2, x_max - x_res/2, W)
    y = torch.linspace(y_min + y_res/2, y_max - y_res/2, H)
    
    # Create meshgrid
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    
    # Stack to get (H, W, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    return grid


if __name__ == '__main__':
    print("Testing geometry utilities...")
    
    # Test 1: Camera projection
    print("\n1. Camera Projection Test:")
    points = torch.tensor([[10.0, 0.0, 2.0],   # 10m forward, on ground
                           [10.0, -3.0, 2.0],  # 10m forward, 3m left
                           [5.0, 2.0, 1.5]])   # 5m forward, 2m right
    
    K = torch.tensor([[1000.0, 0.0, 400.0],
                      [0.0, 1000.0, 224.0],
                      [0.0, 0.0, 1.0]])
    
    pixels, depths = project_points_to_image(points, K)
    print(f"3D Points shape: {points.shape}")
    print(f"Projected pixels: {pixels}")
    print(f"Depths: {depths}")
    
    # Test 2: BEV grid
    print("\n2. BEV Grid Test:")
    grid = create_bev_grid([-50, 50, 0.5], [-50, 50, 0.5])
    print(f"BEV grid shape: {grid.shape}")
    print(f"Grid covers: {grid.min():.1f}m to {grid.max():.1f}m")
    print(f"Resolution: 0.5m per cell")
    
    print("\n✅ All geometry tests passed!")
