import argparse
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate
from utils import set_seed, get_device, log_epoch, BestTracker


def train_one_epoch(model, loader, criterion, optimizer, device):
    """ERM baseline: 仅 CE 损失，不使用敏感属性。"""
    model.train()
    total_loss, n = 0.0, 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device), targets.to(device)
        emb, logits = model(images)
        loss, _, _ = criterion(logits, emb, targets, sensitives=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    args = p.parse_args()

    set_seed()
    device = get_device()
    print(f"Baseline (ERM)  epochs={args.epochs} warmup={args.warmup_epochs}  device={device}")

    train_loader = get_loader("train", args.bs, balanced=False)
    val_loader = get_loader("val", args.bs)

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=0.0)  # 纯 ERM：仅 CE，无 FairSupCon

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=cfg.WD)

    # Warmup + CosineAnnealingWarmRestarts（周期重启，避免 LR 过早衰减到 0）
    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker("baseline(ERM)")
    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        m = evaluate(model, val_loader, device)
        log_epoch(ep, args.epochs, loss, m)
        tracker.update(model, m)

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
