"""
Complete LSS (Lift, Splat, Shoot) Model
Combines: Backbone → Depth → View Transform → BEV
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.depth_net import DepthNet
from src.models.view_transformer import ViewTransformer


class LSSModel(nn.Module):
    """
    Full LSS pipeline for multi-view BEV perception
    
    Architecture:
    1. Image Encoder (ResNet50)
    2. Depth Prediction Network
    3. View Transformer (Lift-Splat-Shoot)
    4. BEV Encoder
    """
    
    def __init__(
        self,
        x_bound=(-50, 50, 0.5),
        y_bound=(-50, 50, 0.5),
        z_bound=(-10, 10, 20),
        d_bound=(4.0, 45.0, 1.0),
        backbone='resnet50',
        bev_channels=64,
    ):
        super().__init__()
        
        # 1. Image Backbone (ResNet50)
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=[3],  # Use layer3 output (1/8 resolution)
        )
        
        # Get backbone output channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 400)
            feats = self.backbone(dummy)
            backbone_channels = feats[0].shape[1]  # Should be 1024 for ResNet50
            feature_size = (feats[0].shape[2], feats[0].shape[3])  # (28, 50)
        
        print(f"Backbone output: {backbone_channels} channels, {feature_size} size")
        
        # 2. Reduce backbone channels
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 3. Depth Prediction Network
        self.depth_net = DepthNet(
            in_channels=512,
            depth_channels=int((d_bound[1] - d_bound[0]) / d_bound[2]),
        )
        
        # 4. Feature compression for voxel pooling
        self.feature_compress = nn.Sequential(
            nn.Conv2d(512, bev_channels, 1),
            nn.ReLU(inplace=True),
        )
        
        # 5. View Transformer
        self.view_transformer = ViewTransformer(
            feature_size=feature_size,
            x_bound=x_bound,
            y_bound=y_bound,
            z_bound=z_bound,
            d_bound=d_bound,
            in_channels=bev_channels,
            out_channels=bev_channels,
        )
        
        # 6. BEV Encoder (process BEV features)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, images, intrinsics, extrinsics):
        """
        Forward pass: Images → BEV features
        
        Args:
            images: (B, N_cam, 3, H, W) multi-view images
            intrinsics: (B, N_cam, 3, 3) camera matrices
            extrinsics: (B, N_cam, 4, 4) camera poses
            
        Returns:
            bev_features: (B, C, bev_h, bev_w) BEV representation
        """
        B, N_cam, C, H, W = images.shape
        
        # 1. Extract features from all cameras
        # Flatten batch and camera dimensions
        images_flat = images.view(B * N_cam, C, H, W)
        
        # Backbone
        feats = self.backbone(images_flat)[0]  # (B*N_cam, 1024, 28, 50)
        
        # Neck
        feats = self.neck(feats)  # (B*N_cam, 512, 28, 50)
        
        # 2. Predict depth
        depth_probs = self.depth_net(feats)  # (B*N_cam, D, 28, 50)
        
        # 3. Compress features
        feats_compressed = self.feature_compress(feats)  # (B*N_cam, 64, 28, 50)
        
        # Reshape back to batch × camera
        _, C_feat, H_feat, W_feat = feats_compressed.shape
        D = depth_probs.shape[1]
        
        feats_compressed = feats_compressed.view(B, N_cam, C_feat, H_feat, W_feat)
        depth_probs = depth_probs.view(B, N_cam, D, H_feat, W_feat)
        
        # 4. Get 3D geometry
        geometry = self.view_transformer.get_geometry(intrinsics, extrinsics)
        
        # 5. Voxel pooling (Lift-Splat-Shoot)
        bev_features = self.view_transformer(geometry, feats_compressed, depth_probs)
        
        # 6. BEV encoding
        bev_features = self.bev_encoder(bev_features)
        
        return bev_features


if __name__ == '__main__':
    print("Testing full LSS model...")
    
    # Create model
    model = LSSModel(
        backbone='resnet50',
        bev_channels=64,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Dummy input
    B, N_cam = 1, 6
    images = torch.randn(B, N_cam, 3, 224, 400)
    intrinsics = torch.randn(B, N_cam, 3, 3)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1)
    
    # Forward pass
    print("\nForward pass...")
    with torch.no_grad():
        bev_features = model(images, intrinsics, extrinsics)
    
    print(f"\nInput: {images.shape}")
    print(f"Output BEV features: {bev_features.shape}")
    print(f"\n✅ Full LSS model working!")
