"""
3D Object Detection Head for BEV Features
Predicts 3D bounding boxes in BEV space
"""
import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    """
    Detect 3D objects from BEV features
    
    For each BEV grid cell, predict:
    - Classification score (10 classes)
    - 3D box center (x, y)
    - Box dimensions (width, length)
    - Orientation (yaw angle)
    """
    
    def __init__(
        self,
        in_channels=64,
        num_classes=10,
        bev_size=(200, 200),
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.bev_size = bev_size
        
        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific heads
        # 1. Classification (which class?)
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        
        # 2. Center offset (x, y offset from grid cell center)
        self.center_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # (x, y)
        )
        
        # 3. Box dimensions (width, length)
        self.dim_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # (w, l)
        )
        
        # 4. Rotation (sin, cos of yaw angle)
        self.rot_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # (sin, cos)
        )
        
    def forward(self, bev_features):
        """
        Args:
            bev_features: (B, C, H, W) BEV representation
            
        Returns:
            predictions: dict with
                - cls: (B, num_classes, H, W) class scores
                - center: (B, 2, H, W) center offsets
                - dim: (B, 2, H, W) box dimensions
                - rot: (B, 2, H, W) rotation (sin, cos)
        """
        # Shared features
        x = self.shared_conv(bev_features)
        
        # Predictions
        cls = self.cls_head(x)
        center = self.center_head(x)
        dim = torch.exp(self.dim_head(x))  # Exp to ensure positive
        rot = self.rot_head(x)
        
        return {
            'cls': cls,
            'center': center,
            'dim': dim,
            'rot': rot,
        }


class CompleteBEVModel(nn.Module):
    """
    Full pipeline: Images → BEV → 3D Detection
    """
    
    def __init__(
        self,
        num_classes=10,
        bev_channels=64,
    ):
        super().__init__()
        
        from src.models.lss import LSSModel
        
        # BEV feature extraction
        self.bev_backbone = LSSModel(bev_channels=bev_channels)
        
        # Detection head
        self.detection_head = BEVDetectionHead(
            in_channels=bev_channels,
            num_classes=num_classes,
        )
        
    def forward(self, images, intrinsics, extrinsics):
        """
        End-to-end: Images → Detections
        """
        # Get BEV features
        bev_features = self.bev_backbone(images, intrinsics, extrinsics)
        
        # Detect objects
        detections = self.detection_head(bev_features)
        
        return detections, bev_features


if __name__ == '__main__':
    print("Testing BEV Detection Head...")
    
    # Test detection head alone
    bev_features = torch.randn(2, 64, 200, 200)
    head = BEVDetectionHead(in_channels=64, num_classes=10)
    
    preds = head(bev_features)
    
    print(f"\nInput BEV: {bev_features.shape}")
    print(f"Output predictions:")
    print(f"  Classification: {preds['cls'].shape}")
    print(f"  Center offset: {preds['center'].shape}")
    print(f"  Dimensions: {preds['dim'].shape}")
    print(f"  Rotation: {preds['rot'].shape}")
    
    # Test complete model
    print("\n" + "="*70)
    print("Testing Complete Model...")
    print("="*70)
    
    model = CompleteBEVModel(num_classes=10, bev_channels=64)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Dummy input
    images = torch.randn(1, 6, 3, 224, 400)
    intrinsics = torch.randn(1, 6, 3, 3)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)
    
    with torch.no_grad():
        detections, bev = model(images, intrinsics, extrinsics)
    
    print(f"\nInput: {images.shape}")
    print(f"BEV features: {bev.shape}")
    print(f"Detections: {detections['cls'].shape}")
    print(f"\n✅ Complete BEV detection model working!")
