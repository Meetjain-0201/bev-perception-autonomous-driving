"""
Test LSS on real nuScenes data
Compare neural BEV vs classical IPM
"""
import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.dataset import NuScenesMultiViewDataset
from src.models.lss import LSSModel
from src.models.ipm import InversePerspectiveMapping

print("="*70)
print("TESTING LSS ON REAL NUSCENES DATA")
print("="*70)

# Load dataset
print("\n1. Loading dataset...")
dataset = NuScenesMultiViewDataset(
    data_root='data/nuscenes',
    version='v1.0-mini',
    split='val',  # Use validation set
    image_size=(224, 400)
)

sample = dataset[0]
print(f"   Loaded sample: {sample['images'].shape}")

# Create LSS model
print("\n2. Creating LSS model...")
model = LSSModel(backbone='resnet50', bev_channels=64)
model.eval()

# Prepare inputs
images = sample['images'].unsqueeze(0)  # (1, 6, 3, 224, 400)
intrinsics = sample['intrinsics'].unsqueeze(0)
extrinsics = sample['extrinsics'].unsqueeze(0)

print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Forward pass
print("\n3. Running LSS forward pass...")
with torch.no_grad():
    bev_features = model(images, intrinsics, extrinsics)

print(f"   Output BEV: {bev_features.shape}")

# Visualize BEV features
print("\n4. Visualizing BEV features...")

# Take mean across channels for visualization
bev_viz = bev_features[0].mean(dim=0).cpu().numpy()  # (200, 200)

# Also get IPM for comparison
print("\n5. Creating IPM comparison...")
ipm = InversePerspectiveMapping()
front_img = sample['images'][0].numpy().transpose(1, 2, 0)
front_img = (front_img * 255).astype(np.uint8)
bev_ipm = ipm.create_bev_from_camera(
    front_img,
    sample['intrinsics'][0].numpy(),
    sample['extrinsics'][0].numpy()
)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Camera image
axes[0].imshow(front_img)
axes[0].set_title('Camera View (CAM_FRONT)', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Classical IPM
axes[1].imshow(bev_ipm, origin='lower')
axes[1].set_title('Classical IPM (Geometric)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('X (lateral, m)')
axes[1].set_ylabel('Y (forward, m)')
axes[1].grid(True, color='yellow', alpha=0.3)

# Neural LSS
im = axes[2].imshow(bev_viz, cmap='viridis', origin='lower')
axes[2].set_title('Neural LSS (Learned)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('X (lateral, m)')
axes[2].set_ylabel('Y (forward, m)')
axes[2].grid(True, color='white', alpha=0.3)
plt.colorbar(im, ax=axes[2], label='Feature Intensity')

plt.suptitle('Comparison: IPM vs LSS BEV Representations', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/images/lss_vs_ipm_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: results/images/lss_vs_ipm_comparison.png")

# Analyze BEV features
print("\n" + "="*70)
print("BEV FEATURE ANALYSIS")
print("="*70)
print(f"BEV shape: {bev_viz.shape}")
print(f"Value range: [{bev_viz.min():.3f}, {bev_viz.max():.3f}]")
print(f"Mean: {bev_viz.mean():.3f}")
print(f"Std: {bev_viz.std():.3f}")
print(f"Non-zero ratio: {(bev_viz != 0).sum() / bev_viz.size:.1%}")

print("\n" + "="*70)
print("✅ LSS TEST COMPLETE!")
print("="*70)
print("\nKey Differences:")
print("  IPM: Direct geometric transform, shows actual pixels")
print("  LSS: Learned features, abstract representation")
print("\nNext: Add detection head to predict 3D bounding boxes from BEV!")
