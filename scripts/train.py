"""
Train the LSS BEV detection model on nuScenes mini.

Run from the project root:
    python scripts/train.py               # 30 epochs (default)
    python scripts/train.py --epochs 5   # quick sanity check
    python scripts/train.py --wandb      # enable W&B logging
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset   import get_dataset, collate_fn
from src.lss_model import LSSModel
from src.loss      import BEVLoss, build_targets
from src.utils     import load_config, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--wandb',  action='store_true')
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, scaler, cfg, device,
              train=True, epoch_idx=0):
    model.train() if train else model.eval()
    mode    = 'Train' if train else 'Val'
    totals  = {'total': 0., 'cls': 0., 'off': 0., 'dim': 0., 'rot': 0.}
    n_batch = 0
    n_total = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs  = batch['images'].to(device)
            Ks    = batch['intrinsics'].to(device)
            exts  = batch['extrinsics'].to(device)
            boxes = batch['boxes']

            targets = build_targets(boxes, cfg, device)

            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds  = model(imgs, Ks, exts)
                losses = criterion(preds, targets)

            if train:
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            for k in totals:
                totals[k] += losses[k].item()
            n_batch += 1

            if train:
                avg = totals['total'] / n_batch
                filled = int(30 * n_batch / n_total)
                bar    = '#' * filled + '-' * (30 - filled)
                print(f'\r    [{bar}] {n_batch:3d}/{n_total}'
                      f'  loss={avg:.4f}'
                      f'  cls={totals["cls"]/n_batch:.4f}'
                      f'  dim={totals["dim"]/n_batch:.4f}',
                      end='', flush=True)

    if train:
        print()

    avg = {k: v / max(n_batch, 1) for k, v in totals.items()}
    print(f'  [{mode}] total={avg["total"]:.4f}'
          f'  cls={avg["cls"]:.4f}  off={avg["off"]:.4f}'
          f'  dim={avg["dim"]:.4f}  rot={avg["rot"]:.4f}')
    return avg


def save_curves(train_hist, val_hist, out_path):
    epochs = range(1, len(train_hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, yscale in zip(axes, ['linear', 'log']):
        ax.plot(epochs, train_hist, label='train', marker='o', markersize=3)
        ax.plot(epochs, val_hist,   label='val',   marker='s', markersize=3)
        ax.set_yscale(yscale)
        ax.set_title(f'Total Loss ({yscale} scale)')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'  Saved loss curves -> {out_path}')


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    n_epochs = args.epochs or cfg['train']['epochs']
    ckpt_dir = cfg['train']['checkpoint_dir']
    res_dir  = cfg['train']['results_dir']
    ensure_dir(ckpt_dir)
    ensure_dir(os.path.join(res_dir, 'images'))

    print('Loading datasets...')
    train_ds = get_dataset('train', cfg)
    val_ds   = get_dataset('val',   cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg['train']['batch_size'],
        shuffle=True,  num_workers=cfg['train']['num_workers'],
        collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds,   batch_size=cfg['train']['batch_size'],
        shuffle=False, num_workers=cfg['train']['num_workers'],
        collate_fn=collate_fn, pin_memory=True)
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    model     = LSSModel(cfg).to(device)
    criterion = BEVLoss(cfg)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['train']['lr'],
                            weight_decay=cfg['train']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda')

    if args.wandb:
        import wandb
        wandb.init(project='bev-perception', config=cfg)

    best_val = float('inf')
    train_hist, val_hist = [], []
    print(f'\nStarting training for {n_epochs} epochs...\n')

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f'Epoch {epoch}/{n_epochs}  lr={scheduler.get_last_lr()[0]:.6f}')

        train_avg = run_epoch(model, train_loader, criterion, optimizer,
                              scaler, cfg, device, train=True)
        val_avg   = run_epoch(model, val_loader,   criterion, optimizer,
                              scaler, cfg, device, train=False)

        scheduler.step()
        train_hist.append(train_avg['total'])
        val_hist.append(val_avg['total'])

        if args.wandb:
            import wandb
            wandb.log({'train/total': train_avg['total'],
                       'val/total':   val_avg['total'],
                       'lr': scheduler.get_last_lr()[0]}, step=epoch)

        if val_avg['total'] < best_val:
            best_val  = val_avg['total']
            ckpt_path = os.path.join(ckpt_dir, 'best.pth')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'opt_state':   optimizer.state_dict(),
                'val_loss':    best_val,
                'cfg':         cfg,
            }, ckpt_path)
            print(f'  ** New best val={best_val:.4f} -> {ckpt_path}')

        print(f'  Epoch time: {time.time() - t0:.1f}s\n')
        save_curves(train_hist, val_hist,
                    os.path.join(res_dir, 'images', 'loss_curves.png'))

    print(f'Training complete. Best val loss: {best_val:.4f}')
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
