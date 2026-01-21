"""
Loss functions for BEV 3D object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDetectionLoss(nn.Module):
    """
    Multi-task loss for BEV detection
    
    Components:
    1. Classification loss (focal loss for class imbalance)
    2. Center regression loss (L1)
    3. Dimension regression loss (L1)
    4. Rotation loss (smooth L1)
    """
    
    def __init__(
        self,
        num_classes=10,
        cls_weight=1.0,
        center_weight=1.0,
        dim_weight=1.0,
        rot_weight=0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.center_weight = center_weight
        self.dim_weight = dim_weight
        self.rot_weight = rot_weight
        
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal loss for classification
        Handles class imbalance (most BEV cells are empty)
        """
        # Sigmoid
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal weight
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        # Apply focal weight
        loss = alpha * focal_weight * bce_loss
        
        return loss.mean()
    
    def forward(self, predictions, targets):
        """
        Compute total loss
        
        Args:
            predictions: dict with 'cls', 'center', 'dim', 'rot'
            targets: dict with same keys
            
        Returns:
            loss: scalar tensor
            loss_dict: dict with individual losses
        """
        # 1. Classification loss
        cls_loss = self.focal_loss(
            predictions['cls'],
            targets['cls']
        )
        
        # 2. Center regression (only where objects exist)
        mask = targets['cls'].max(dim=1, keepdim=True)[0] > 0.5  # (B, 1, H, W)
        
        if mask.sum() > 0:
            center_loss = F.l1_loss(
                predictions['center'] * mask,
                targets['center'] * mask,
                reduction='sum'
            ) / (mask.sum() + 1e-6)
            
            dim_loss = F.l1_loss(
                predictions['dim'] * mask,
                targets['dim'] * mask,
                reduction='sum'
            ) / (mask.sum() + 1e-6)
            
            rot_loss = F.smooth_l1_loss(
                predictions['rot'] * mask,
                targets['rot'] * mask,
                reduction='sum'
            ) / (mask.sum() + 1e-6)
        else:
            center_loss = torch.tensor(0.0, device=predictions['cls'].device)
            dim_loss = torch.tensor(0.0, device=predictions['cls'].device)
            rot_loss = torch.tensor(0.0, device=predictions['cls'].device)
        
        # Total loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.center_weight * center_loss +
            self.dim_weight * dim_loss +
            self.rot_weight * rot_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'center': center_loss.item(),
            'dim': dim_loss.item(),
            'rot': rot_loss.item(),
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    print("Testing detection loss...")
    
    # Dummy predictions and targets
    B, C, H, W = 2, 10, 200, 200
    
    predictions = {
        'cls': torch.randn(B, C, H, W),
        'center': torch.randn(B, 2, H, W),
        'dim': torch.randn(B, 2, H, W),
        'rot': torch.randn(B, 2, H, W),
    }
    
    # Create dummy targets (sparse - few objects)
    targets = {
        'cls': torch.zeros(B, C, H, W),
        'center': torch.zeros(B, 2, H, W),
        'dim': torch.zeros(B, 2, H, W),
        'rot': torch.zeros(B, 2, H, W),
    }
    
    # Add one object
    targets['cls'][0, 0, 100, 100] = 1  # Car at center
    targets['center'][0, :, 100, 100] = torch.tensor([0.5, 0.3])
    targets['dim'][0, :, 100, 100] = torch.tensor([4.0, 2.0])
    
    # Compute loss
    criterion = BEVDetectionLoss()
    loss, loss_dict = criterion(predictions, targets)
    
    print(f"\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k:10s}: {v:.4f}")
    
    print(f"\n✅ Loss functions working!")
