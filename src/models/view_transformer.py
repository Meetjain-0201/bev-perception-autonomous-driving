"""
View Transformer for LSS (Lift, Splat, Shoot)
This is where the magic happens!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewTransformer(nn.Module):
    """
    Transforms image features to BEV using depth prediction
    
    Steps:
    1. LIFT: Use depth to lift 2D features to 3D frustum
    2. SPLAT: Scatter 3D points into voxel grid
    3. SHOOT: Collapse to BEV
    """
    
    def __init__(
        self,
        image_size=(224, 400),
        feature_size=(28, 50),  # After backbone downsampling (8x)
        x_bound=(-50, 50, 0.5),
        y_bound=(-50, 50, 0.5),
        z_bound=(-10, 10, 20),
        d_bound=(4.0, 45.0, 1.0),
        in_channels=64,
        out_channels=64,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.feature_size = feature_size
        
        # BEV grid parameters
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        
        # Calculate grid dimensions
        self.bev_x = int((x_bound[1] - x_bound[0]) / x_bound[2])  # 200
        self.bev_y = int((y_bound[1] - y_bound[0]) / y_bound[2])  # 200
        self.bev_z = int(z_bound[2])  # 20 height bins
        
        # Depth bins
        self.num_depth_bins = int((d_bound[1] - d_bound[0]) / d_bound[2])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create frustum (3D grid for each pixel)
        self.frustum = self.create_frustum()
        
    def create_frustum(self):
        """
        Create frustum: 3D points for each pixel at each depth
        
        Returns:
            frustum: (D, H_feat, W_feat, 3) coordinates in camera frame
                     where D = num_depth_bins
        """
        # Depth samples
        d_min, d_max, d_step = self.d_bound
        depths = torch.arange(d_min, d_max, d_step)  # (D,)
        
        H_feat, W_feat = self.feature_size
        
        # Pixel coordinates (downsampled)
        h_coords = torch.linspace(0, self.image_size[0] - 1, H_feat)
        w_coords = torch.linspace(0, self.image_size[1] - 1, W_feat)
        
        # Meshgrid
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing='ij')
        
        # Expand for depth dimension
        grid_h = grid_h[None, ...].expand(len(depths), -1, -1)
        grid_w = grid_w[None, ...].expand(len(depths), -1, -1)
        depths_grid = depths[:, None, None].expand(-1, H_feat, W_feat)
        
        # Stack: (D, H, W, 3) where 3 = [u, v, depth]
        frustum = torch.stack([grid_w, grid_h, depths_grid], dim=-1)
        
        return frustum
    
    def get_geometry(self, intrinsics, extrinsics):
        """
        Get 3D coordinates in ego frame for each pixel-depth combination
        
        Args:
            intrinsics: (B, N_cam, 3, 3) camera matrices
            extrinsics: (B, N_cam, 4, 4) camera poses
            
        Returns:
            geometry: (B, N_cam, D, H, W, 3) 3D coordinates in ego frame
        """
        B, N_cam, _, _ = intrinsics.shape
        D, H_feat, W_feat, _ = self.frustum.shape
        
        # Move frustum to device
        frustum = self.frustum.to(intrinsics.device)  # (D, H, W, 3)
        
        geometry_list = []
        
        for b in range(B):
            batch_geometry = []
            
            for cam in range(N_cam):
                K = intrinsics[b, cam]  # (3, 3)
                cam_to_ego = extrinsics[b, cam]  # (4, 4)
                
                # Unproject frustum to 3D camera frame
                # frustum has [u, v, d]
                # We want [X, Y, Z] in camera frame
                
                u = frustum[..., 0]  # (D, H, W)
                v = frustum[..., 1]  # (D, H, W)
                d = frustum[..., 2]  # (D, H, W)
                
                # Inverse projection
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                
                # 3D point in camera frame
                X = (u - cx) * d / fx
                Y = (v - cy) * d / fy
                Z = d
                
                # Stack to (D, H, W, 3)
                points_cam = torch.stack([X, Y, Z], dim=-1)
                
                # Transform to ego frame
                # points_ego = cam_to_ego @ [X, Y, Z, 1]
                points_cam_flat = points_cam.reshape(-1, 3)  # (D*H*W, 3)
                
                # Add homogeneous coordinate
                ones = torch.ones(points_cam_flat.shape[0], 1, device=points_cam_flat.device)
                points_cam_homo = torch.cat([points_cam_flat, ones], dim=1)  # (N, 4)
                
                # Transform
                points_ego_homo = (cam_to_ego @ points_cam_homo.T).T  # (N, 4)
                points_ego = points_ego_homo[:, :3]  # (N, 3)
                
                # Reshape back
                points_ego = points_ego.reshape(D, H_feat, W_feat, 3)
                
                batch_geometry.append(points_ego)
            
            # Stack cameras
            batch_geometry = torch.stack(batch_geometry, dim=0)  # (N_cam, D, H, W, 3)
            geometry_list.append(batch_geometry)
        
        # Stack batches
        geometry = torch.stack(geometry_list, dim=0)  # (B, N_cam, D, H, W, 3)
        
        return geometry
    
    def voxel_pooling(self, geometry, features, depth_probs):
        """
        Pool features into voxel grid (the "Splat" operation)
        
        Args:
            geometry: (B, N_cam, D, H, W, 3) 3D coordinates
            features: (B, N_cam, C, H, W) image features
            depth_probs: (B, N_cam, D, H, W) depth probabilities
            
        Returns:
            bev_features: (B, C, bev_x, bev_y) BEV features
        """
        B, N_cam, D, H, W, _ = geometry.shape
        C = features.shape[2]
        
        # Initialize voxel grid
        voxel_grid = torch.zeros(
            B, C, self.bev_z, self.bev_y, self.bev_x,
            device=features.device
        )
        
        # For each camera
        for b in range(B):
            for cam in range(N_cam):
                # Get this camera's data
                geom = geometry[b, cam]  # (D, H, W, 3)
                feat = features[b, cam]  # (C, H, W)
                depth = depth_probs[b, cam]  # (D, H, W)
                
                # Flatten
                geom_flat = geom.reshape(-1, 3)  # (D*H*W, 3)
                
                # Repeat features for each depth
                feat_expanded = feat.unsqueeze(1).expand(-1, D, -1, -1)  # (C, D, H, W)
                feat_flat = feat_expanded.reshape(C, -1)  # (C, D*H*W)
                
                # Flatten depth weights
                depth_flat = depth.reshape(-1)  # (D*H*W,)
                
                # Find voxel indices
                x_coords = geom_flat[:, 0]
                y_coords = geom_flat[:, 1]
                z_coords = geom_flat[:, 2]
                
                # Discretize to voxel grid
                x_idx = ((x_coords - self.x_bound[0]) / self.x_bound[2]).long()
                y_idx = ((y_coords - self.y_bound[0]) / self.y_bound[2]).long()
                z_idx = ((z_coords - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0]) * self.bev_z).long()
                
                # Filter valid indices
                valid = (
                    (x_idx >= 0) & (x_idx < self.bev_x) &
                    (y_idx >= 0) & (y_idx < self.bev_y) &
                    (z_idx >= 0) & (z_idx < self.bev_z)
                )
                
                # Accumulate features into voxels (weighted by depth probability)
                for c in range(C):
                    for i in range(len(x_idx)):
                        if valid[i]:
                            voxel_grid[b, c, z_idx[i], y_idx[i], x_idx[i]] += \
                                feat_flat[c, i] * depth_flat[i]
        
        # Collapse Z dimension (max pooling)
        bev_features = voxel_grid.max(dim=2)[0]  # (B, C, bev_y, bev_x)
        
        return bev_features
    
    def forward(self, geometry, features, depth_probs):
        """
        Full forward pass
        
        Args:
            geometry: (B, N_cam, D, H, W, 3)
            features: (B, N_cam, C, H, W)
            depth_probs: (B, N_cam, D, H, W)
            
        Returns:
            bev_features: (B, C, bev_y, bev_x)
        """
        bev_features = self.voxel_pooling(geometry, features, depth_probs)
        return bev_features


if __name__ == '__main__':
    print("Testing ViewTransformer...")
    
    # Create transformer
    transformer = ViewTransformer(
        feature_size=(28, 50),
        in_channels=64,
        out_channels=64,
    )
    
    # Dummy data
    B, N_cam = 2, 6
    C, H, W = 64, 28, 50
    D = 112
    
    # Create dummy geometry
    intrinsics = torch.randn(B, N_cam, 3, 3)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1)
    
    geometry = transformer.get_geometry(intrinsics, extrinsics)
    print(f"Geometry: {geometry.shape}")
    
    # Dummy features and depth
    features = torch.randn(B, N_cam, C, H, W)
    depth_probs = F.softmax(torch.randn(B, N_cam, D, H, W), dim=2)
    
    # Forward
    bev = transformer(geometry, features, depth_probs)
    
    print(f"Input features: {features.shape}")
    print(f"Output BEV: {bev.shape}")
    print(f"\n✅ ViewTransformer working!")
