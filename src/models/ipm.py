"""
Classical Inverse Perspective Mapping (IPM)
Geometric transformation from camera view to Bird's Eye View
"""
import torch
import torch.nn as nn
import numpy as np
import cv2


class InversePerspectiveMapping:
    """
    Classical IPM using homography transformation
    
    Assumptions:
    - Ground plane is flat (Z = 0)
    - Camera calibration is known
    - Static scene (works for road, not for 3D objects)
    """
    
    def __init__(
        self,
        image_size=(224, 400),
        bev_size=(200, 200),
        bev_range=(-50, 50, -50, 50),  # (x_min, x_max, y_min, y_max) in meters
    ):
        """
        Args:
            image_size: (H, W) of input images
            bev_size: (H, W) of output BEV grid
            bev_range: (x_min, x_max, y_min, y_max) BEV coverage in meters
        """
        self.image_h, self.image_w = image_size
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max, self.y_min, self.y_max = bev_range
        
        # BEV resolution (meters per pixel)
        self.x_res = (self.x_max - self.x_min) / self.bev_w
        self.y_res = (self.y_max - self.y_min) / self.bev_h
        
    def compute_homography(self, intrinsics, extrinsics):
        """
        Compute homography matrix for ground plane transformation
        
        Theory:
        - Camera sees 3D world, projects to 2D image
        - For ground plane (Z=0), we can invert this projection
        - Result: direct mapping from image pixels → ground coordinates
        
        Args:
            intrinsics: (3, 3) camera intrinsic matrix K
            extrinsics: (4, 4) camera-to-ego transformation
            
        Returns:
            H: (3, 3) homography matrix
        """
        # Extract rotation and translation
        R = extrinsics[:3, :3]  # Camera rotation
        t = extrinsics[:3, 3]   # Camera translation
        
        # Homography for ground plane (Z=0):
        # H = K × [R₁ R₂ t]
        # where R₁, R₂ are first two columns of rotation matrix
        
        # Build homography
        H = intrinsics @ np.hstack([R[:, [0, 1]], t.reshape(3, 1)])
        
        return H
    
    def transform_image_to_bev(self, image, intrinsics, extrinsics):
        """
        Transform camera image to BEV using IPM
        
        Args:
            image: (H, W, 3) numpy array, camera image
            intrinsics: (3, 3) camera matrix
            extrinsics: (4, 4) camera pose
            
        Returns:
            bev_image: (BEV_H, BEV_W, 3) transformed to BEV
        """
        # Ensure numpy arrays
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        
        # Compute homography
        H = self.compute_homography(intrinsics, extrinsics)
        
        # Invert homography (BEV → image)
        H_inv = np.linalg.inv(H)
        
        # Create BEV grid coordinates
        bev_x = np.linspace(self.x_min, self.x_max, self.bev_w)
        bev_y = np.linspace(self.y_min, self.y_max, self.bev_h)
        
        # Meshgrid of BEV coordinates
        grid_x, grid_y = np.meshgrid(bev_x, bev_y)
        
        # Flatten and convert to homogeneous coordinates
        ones = np.ones_like(grid_x.flatten())
        bev_coords = np.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            ones
        ], axis=0)  # (3, N)
        
        # Transform BEV → image pixels
        image_coords = H_inv @ bev_coords  # (3, N)
        
        # Normalize homogeneous coordinates
        image_coords = image_coords[:2, :] / (image_coords[2:, :] + 1e-6)
        
        # Reshape to grid
        u = image_coords[0, :].reshape(self.bev_h, self.bev_w)
        v = image_coords[1, :].reshape(self.bev_h, self.bev_w)
        
        # Sample from image using remap
        bev_image = cv2.remap(
            image,
            u.astype(np.float32),
            v.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return bev_image
    
    def visualize_transformation(self, image, bev_image, save_path=None):
        """
        Visualize side-by-side: camera view vs BEV
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Camera view
        axes[0].imshow(image)
        axes[0].set_title('Camera View (Perspective)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # BEV view
        axes[1].imshow(bev_image)
        axes[1].set_title('Bird\'s Eye View (IPM)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (lateral) →')
        axes[1].set_ylabel('Y (longitudinal) →')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.close()


if __name__ == '__main__':
    print("Testing IPM implementation...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (224, 400, 3), dtype=np.uint8)
    K = np.array([[1000, 0, 200], [0, 1000, 112], [0, 0, 1]], dtype=np.float32)
    
    # Camera 1.5m high, looking forward and down
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[2, 3] = 1.5  # Height
    
    # Create IPM
    ipm = InversePerspectiveMapping(
        image_size=(224, 400),
        bev_size=(200, 200),
        bev_range=(-25, 25, 0, 50)  # 50m forward
    )
    
    # Transform
    H = ipm.compute_homography(K, extrinsics)
    print(f"Homography matrix shape: {H.shape}")
    print(f"Homography:\n{H}")
    
    print("\n✅ IPM implementation ready!")
