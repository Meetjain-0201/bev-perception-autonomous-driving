"""
Check how many objects are in BEV range across samples
"""
import sys
sys.path.append('.')

from src.data.dataset import NuScenesMultiViewDataset
import numpy as np

dataset = NuScenesMultiViewDataset(
    data_root='data/nuscenes',
    version='v1.0-mini',
    split='train',
    return_targets=True
)

print(f"Checking {len(dataset)} samples...")
print("\nSample\tObjects")
print("-" * 30)

object_counts = []
for idx in range(min(20, len(dataset))):  # Check first 20
    sample = dataset[idx]
    num_objects = sample['targets']['cls'].sum().item()
    object_counts.append(num_objects)
    if idx < 10:
        print(f"{idx:3d}\t{num_objects:.0f}")

print(f"\nStatistics:")
print(f"  Mean objects per sample: {np.mean(object_counts):.1f}")
print(f"  Max objects: {np.max(object_counts):.0f}")
print(f"  Samples with objects: {np.sum(np.array(object_counts) > 0)}/{len(object_counts)}")

# Find a good sample with objects
good_samples = [i for i, c in enumerate(object_counts) if c > 0]
if good_samples:
    print(f"\n✅ Good samples with objects: {good_samples[:5]}")
else:
    print(f"\n⚠️  No objects found in BEV range - may need to adjust bounds")
