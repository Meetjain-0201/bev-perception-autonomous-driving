"""
Test complete BEV detection pipeline
"""
import sys
sys.path.append('.')

import torch
from src.models.detection_head import BEVDetectionHead, CompleteBEVModel

print("="*70)
print("TESTING BEV DETECTION HEAD")
print("="*70)

# Test 1: Detection head alone
print("\n1. Testing Detection Head...")
bev_features = torch.randn(2, 64, 200, 200)
head = BEVDetectionHead(in_channels=64, num_classes=10)

preds = head(bev_features)

print(f"Input BEV: {bev_features.shape}")
print(f"\nOutput predictions:")
print(f"  Classification: {preds['cls'].shape}")
print(f"  Center offset: {preds['center'].shape}")
print(f"  Dimensions: {preds['dim'].shape}")
print(f"  Rotation: {preds['rot'].shape}")

# Test 2: Complete model
print("\n" + "="*70)
print("TESTING COMPLETE END-TO-END MODEL")
print("="*70)

model = CompleteBEVModel(num_classes=10, bev_channels=64)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Dummy input
images = torch.randn(1, 6, 3, 224, 400)
intrinsics = torch.randn(1, 6, 3, 3)
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)

print("\nRunning end-to-end forward pass...")
with torch.no_grad():
    detections, bev = model(images, intrinsics, extrinsics)

print(f"\nPipeline:")
print(f"  Input images: {images.shape}")
print(f"  ↓")
print(f"  BEV features: {bev.shape}")
print(f"  ↓")
print(f"  Detections: {detections['cls'].shape}")

print("\n" + "="*70)
print("✅ COMPLETE MODEL WORKING!")
print("="*70)

print("\n📊 Model Summary:")
print(f"  Input: 6 cameras × 224×400 RGB")
print(f"  Backbone: ResNet50 (pretrained)")
print(f"  BEV: 200×200 grid, 64 channels")
print(f"  Output: 3D boxes (class, center, size, rotation)")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

print("\n🎯 Next Steps:")
print("  1. Training pipeline (loss functions, optimizer)")
print("  2. Evaluation metrics (mAP, NDS)")
print("  3. Visualization of predictions")
