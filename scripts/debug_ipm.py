"""
Debug IPM - Fixed for WSL2
"""
import sys
sys.path.append('.')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from src.data.dataset import NuScenesMultiViewDataset
from src.models.ipm import InversePerspectiveMapping

# Load data
print("Loading dataset...")
dataset = NuScenesMultiViewDataset(
    data_root='data/nuscenes',
    version='v1.0-mini',
    split='train',
    image_size=(224, 400)
)

sample = dataset[0]

# Get front camera
front_idx = 0
image = sample['images'][front_idx]
K = sample['intrinsics'][front_idx]
extrinsics = sample['extrinsics'][front_idx]

print(f"\nCamera Analysis:")
print(f"  Image shape: {image.shape}")
print(f"  Camera height: {extrinsics[2, 3]:.2f}m")
print(f"  Camera translation: [{extrinsics[0,3]:.2f}, {extrinsics[1,3]:.2f}, {extrinsics[2,3]:.2f}]")

# Convert for IPM
img_np = image.numpy().transpose(1, 2, 0)
img_np = (img_np * 255).astype(np.uint8)

# Create IPM
ipm = InversePerspectiveMapping(
    image_size=(224, 400),
    bev_size=(200, 200),
    bev_range=(-25, 25, 5, 50)
)

# Transform
print("\nApplying IPM...")
bev = ipm.create_bev_from_camera(img_np, K.numpy(), extrinsics.numpy())

print(f"\nBEV Statistics:")
print(f"  Shape: {bev.shape}")
print(f"  Non-zero pixels: {np.count_nonzero(bev)}/{bev.shape[0]*bev.shape[1]}")
print(f"  Coverage: {100*np.count_nonzero(bev)/(bev.shape[0]*bev.shape[1]):.1f}%")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original image
axes[0].imshow(img_np)
axes[0].set_title('Camera View (CAM_FRONT)', fontsize=12, fontweight='bold')
axes[0].axis('off')

# BEV
axes[1].imshow(bev, origin='lower')
axes[1].set_title('BEV (IPM Transform)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('X (lateral, m)')
axes[1].set_ylabel('Y (forward, m)')

# BEV with grid
axes[2].imshow(bev, origin='lower')
axes[2].set_title('BEV with Coverage Grid', fontsize=12, fontweight='bold')
axes[2].grid(True, color='yellow', alpha=0.5, linewidth=0.5)
axes[2].set_xlabel('X (lateral, m)')
axes[2].set_ylabel('Y (forward, m)')

plt.tight_layout()
plt.savefig('results/images/debug_ipm_detailed.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: results/images/debug_ipm_detailed.png")

# Test on multiple samples
print("\n" + "="*60)
print("Testing on 5 different samples:")
print("="*60)

for idx in [0, 50, 100, 150, 200]:
    sample = dataset[idx]
    img = sample['images'][0].numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    bev = ipm.create_bev_from_camera(
        img, 
        sample['intrinsics'][0].numpy(),
        sample['extrinsics'][0].numpy()
    )
    coverage = 100 * np.count_nonzero(bev) / (bev.shape[0] * bev.shape[1])
    print(f"  Sample {idx:3d}: {coverage:5.1f}% coverage")

print("\n✅ Debug complete! Check results/images/debug_ipm_detailed.png")
