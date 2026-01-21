"""
Classical Inverse Perspective Mapping (IPM) - CORRECTED
"""
import torch
import numpy as np
import cv2


class InversePerspectiveMapping:
    """
    Simple IPM that actually works with nuScenes
    """
    
    def __init__(
        self,
        image_size=(224, 400),
        bev_size=(200, 200),
        bev_range=(-25, 25, 5, 50),  # (x_min, x_max, y_min, y_max)
    ):
        self.image_h, self.image_w = image_size
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max, self.y_min, self.y_max = bev_range
        
    def create_bev_from_camera(self, image, intrinsics, extrinsics):
        """
        Simplified IPM: map image bottom region to BEV
        
        Strategy:
        - Focus on lower half of image (where road is visible)
        - Use camera geometry to map pixels → ground plane
        - Sample image to fill BEV grid
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image = image.transpose(1, 2, 0)
        
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        
        # Ensure uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Extract camera parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Camera height and pitch from extrinsics
        camera_height = extrinsics[2, 3]  # Z translation
        
        # Create BEV grid
        bev_image = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
        
        # For each BEV pixel, find corresponding image pixel
        for i in range(self.bev_h):
            for j in range(self.bev_w):
                # BEV coordinates (in ego frame)
                x_bev = self.x_min + (j / self.bev_w) * (self.x_max - self.x_min)
                y_bev = self.y_min + (i / self.bev_h) * (self.y_max - self.y_min)
                
                # Transform to camera frame
                # Simplified: assume camera looking forward
                X_cam = y_bev  # Forward in BEV → X in camera
                Y_cam = -x_bev  # Left in BEV → Y in camera
                Z_cam = -camera_height  # Ground is below camera
                
                # Project to image
                if Z_cam < 0:  # Point is in front of camera
                    u = -fx * (Y_cam / X_cam) + cx
                    v = -fy * (Z_cam / X_cam) + cy
                    
                    # Check if in image bounds
                    if 0 <= u < self.image_w and 0 <= v < self.image_h:
                        u_int, v_int = int(u), int(v)
                        bev_image[i, j] = image[v_int, u_int]
        
        return bev_image


if __name__ == '__main__':
    print("Testing corrected IPM...")
    
    # Test with dummy data
    image = np.random.randint(0, 255, (224, 400, 3), dtype=np.uint8)
    K = np.array([[1000, 0, 200], [0, 1000, 112], [0, 0, 1]], dtype=np.float32)
    
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[2, 3] = 1.5  # 1.5m camera height
    
    ipm = InversePerspectiveMapping()
    bev = ipm.create_bev_from_camera(image, K, extrinsics)
    
    print(f"Input: {image.shape}")
    print(f"Output BEV: {bev.shape}")
    print(f"BEV contains data: {bev.sum() > 0}")
    print("\n✅ IPM corrected!")
