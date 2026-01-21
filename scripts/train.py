"""
Training script for BEV perception
Optimized for CPU training
"""
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from src.data.dataset import NuScenesMultiViewDataset
from src.models.detection_head import CompleteBEVModel
from src.losses.detection_loss import BEVDetectionLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    
    epoch_losses = {'total': 0, 'cls': 0, 'center': 0, 'dim': 0, 'rot': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        targets = {k: v.to(device) for k, v in batch['targets'].items()}
        
        # Forward
        optimizer.zero_grad()
        detections, bev_features = model(images, intrinsics, extrinsics)
        
        # Loss
        loss, loss_dict = criterion(detections, targets)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Accumulate
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        
        # Update progress
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{loss_dict['cls']:.4f}",
        })
    
    # Average
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    return avg_losses


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    
    val_losses = {'total': 0, 'cls': 0, 'center': 0, 'dim': 0, 'rot': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            detections, _ = model(images, intrinsics, extrinsics)
            loss, loss_dict = criterion(detections, targets)
            
            for k, v in loss_dict.items():
                val_losses[k] += v
    
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in val_losses.items()}
    
    return avg_losses


def main():
    print("="*70)
    print("BEV PERCEPTION TRAINING")
    print("="*70)
    
    # Config
    device = torch.device('cpu')
    batch_size = 1  # CPU limitation
    num_epochs = 15
    learning_rate = 2e-4
    
    # Dataset
    print("\n📊 Loading dataset...")
    train_dataset = NuScenesMultiViewDataset(
        data_root='data/nuscenes',
        version='v1.0-mini',
        split='train',
        return_targets=True
    )
    
    val_dataset = NuScenesMultiViewDataset(
        data_root='data/nuscenes',
        version='v1.0-mini',
        split='val',
        return_targets=True
    )
    
    # Use subset for CPU (50 train, 10 val)
    train_subset = torch.utils.data.Subset(train_dataset, range(50))
    val_subset = torch.utils.data.Subset(val_dataset, range(10))
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for CPU
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"   Train: {len(train_subset)} samples")
    print(f"   Val: {len(val_subset)} samples")
    
    # Model
    print("\n🏗️  Creating model...")
    model = CompleteBEVModel(num_classes=10, bev_channels=64)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e6:.2f}M")
    
    # Loss & Optimizer
    criterion = BEVDetectionLoss(
        cls_weight=1.0,
        center_weight=2.0,
        dim_weight=1.0,
        rot_weight=0.5
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training
    print("\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Estimated time: ~{num_epochs * 10} min on CPU")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_losses = validate(model, val_loader, criterion, device)
            val_history.append(val_losses)
            
            print(f"\n📊 Epoch {epoch}/{num_epochs} ({time.time()-epoch_start:.1f}s):")
            print(f"   Train - Loss: {train_losses['total']:.4f} | "
                  f"Cls: {train_losses['cls']:.4f} | "
                  f"Reg: {train_losses['center']:.4f}")
            print(f"   Val   - Loss: {val_losses['total']:.4f} | "
                  f"Cls: {val_losses['cls']:.4f} | "
                  f"Reg: {val_losses['center']:.4f}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }
                torch.save(checkpoint, 'checkpoints/best_model.pth')
                print(f"   ✅ Best model saved!")
        else:
            print(f"\n📊 Epoch {epoch}/{num_epochs} ({time.time()-epoch_start:.1f}s):")
            print(f"   Train - Loss: {train_losses['total']:.4f} | "
                  f"Cls: {train_losses['cls']:.4f}")
        
        train_history.append(train_losses)
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    
    # Save training history
    import json
    with open('logs/training_history.json', 'w') as f:
        json.dump({
            'train': train_history,
            'val': val_history
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved models:")
    print(f"  - checkpoints/best_model.pth")
    print(f"  - checkpoints/final_model.pth")
    print(f"  - logs/training_history.json")


if __name__ == '__main__':
    main()
