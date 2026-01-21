"""
Depth Prediction Network for LSS
Predicts depth distribution for each pixel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):
    """
    Predicts depth distribution for each image pixel
    
    Uses categorical depth prediction:
    - Divide depth range into D bins (e.g., 4m to 45m, 112 bins)
    - Predict probability distribution over bins for each pixel
    - Output: (B, D, H, W) depth probabilities
    """
    
    def __init__(
        self,
        in_channels=512,  # From backbone features
        depth_channels=112,  # Number of depth bins
        mid_channels=256,
    ):
        super().__init__()
        
        self.depth_channels = depth_channels
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, depth_channels, 1),  # 1x1 conv
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) image features from backbone
            
        Returns:
            depth_probs: (B, D, H, W) depth probability distribution
        """
        # Predict depth logits
        depth_logits = self.depth_head(features)  # (B, D, H, W)
        
        # Softmax over depth dimension
        depth_probs = F.softmax(depth_logits, dim=1)
        
        return depth_probs


if __name__ == '__main__':
    print("Testing DepthNet...")
    
    # Dummy input (backbone features)
    B, C, H, W = 2, 512, 28, 50
    features = torch.randn(B, C, H, W)
    
    # Create depth net
    depth_net = DepthNet(
        in_channels=512,
        depth_channels=112
    )
    
    # Forward pass
    depth_probs = depth_net(features)
    
    print(f"Input features: {features.shape}")
    print(f"Output depth probs: {depth_probs.shape}")
    print(f"Depth probs sum to 1: {depth_probs[0, :, 0, 0].sum():.3f}")
    print(f"\n✅ DepthNet working!")
