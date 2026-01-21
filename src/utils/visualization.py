"""
Visualization utilities for BEV perception
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for WSL2
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple


def plot_multiview_cameras(images: np.ndarray, camera_names: List[str], 
                           save_path: str = None):
    """
    Plot 6-camera surround view
    
    Args:
        images: (6, H, W, 3) numpy array
        camera_names: List of 6 camera names
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    for idx, (img, name) in enumerate(zip(images, camera_names)):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.close()
    

def plot_bev_grid(bev_features: np.ndarray, title: str = "BEV Grid",
                  save_path: str = None):
    """
    Visualize BEV grid representation
    
    Args:
        bev_features: (H, W) or (H, W, C) BEV grid
        title: Plot title
        save_path: Optional path to save
    """
    if len(bev_features.shape) == 3:
        # Take mean across channels for visualization
        bev_features = bev_features.mean(axis=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_features, cmap='viridis', origin='lower')
    plt.colorbar(label='Feature Intensity')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X (lateral, meters)')
    plt.ylabel('Y (longitudinal, meters)')
    
    # Add grid
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.close()


def draw_boxes_on_image(image: np.ndarray, boxes_2d: List[Tuple], 
                        labels: List[str] = None, color=(0, 255, 0)):
    """
    Draw 2D bounding boxes on image
    
    Args:
        image: (H, W, 3) image
        boxes_2d: List of (x1, y1, x2, y2) boxes
        labels: Optional class labels
        color: Box color in BGR
    """
    img_draw = image.copy()
    
    for idx, box in enumerate(boxes_2d):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        if labels and idx < len(labels):
            cv2.putText(img_draw, labels[idx], (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_draw


if __name__ == '__main__':
    print("Testing visualization utilities...")
    
    # Test BEV grid visualization
    dummy_bev = np.random.rand(200, 200)
    save_path = 'results/images/test_bev_grid.png'
    plot_bev_grid(dummy_bev, title="Test BEV Grid", save_path=save_path)
    
    print("✅ Visualization utils ready!")
