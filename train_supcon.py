import argparse
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier
from loss import TotalLoss
from eval import evaluate
from utils import set_seed, get_device, log_epoch, BestTracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--balanced", action="store_true", help="Use group-balanced sampling")
    p.add_argument("--unbalanced", action="store_true", help="Use unbalanced sampling")
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS, help=f"Number of epochs (default: {cfg.NUM_EPOCHS})")
    p.add_argument("--lambda-con", type=float, default=cfg.LAMBDA_CON, help=f"Contrastive loss weight (default: {cfg.LAMBDA_CON})")
    p.add_argument("--temperature", type=float, default=cfg.TEMPERATURE, help=f"SupCon temperature tau (default: {cfg.TEMPERATURE})")
    args = p.parse_args()

    if args.balanced and args.unbalanced:
        raise ValueError("Cannot specify both --balanced and --unbalanced")

    # 默认balanced平衡采样
    # 有 --balanced 则平衡采样，有 --unbalanced 则均匀采样
    balanced = args.balanced if args.balanced else (not args.unbalanced)

    set_seed()
    device = get_device()
    tag = "FSC_balanced_"+cfg.TARGET_ATTR+"_vs_"+cfg.SENSITIVE_ATTR if balanced else "FSC_unbalanced_"+cfg.TARGET_ATTR+"_vs_"+cfg.SENSITIVE_ATTR
    lambda_con = args.lambda_con
    temperature = args.temperature
    epochs = args.epochs
    print(f"Fairness Sup_Con:  epochs={epochs}  lambda={lambda_con}  tau={temperature}  balanced={balanced}  warmup={cfg.WARMUP_EPOCHS}  device={device}")

    train_loader = get_loader("train", cfg.BATCH_SIZE, balanced=balanced)
    val_loader = get_loader("val", cfg.BATCH_SIZE)

    model = FairClassifier().to(device)
    criterion = TotalLoss(lambda_con=lambda_con, temperature=temperature)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR},
    ], weight_decay=cfg.WD)

    sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warmup, sched_cosine], milestones=[cfg.WARMUP_EPOCHS])

    tracker = BestTracker(tag, warmup_epochs=cfg.WARMUP_EPOCHS)
    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets, sensitives, _ in train_loader:
            imgs, tgts, sens = images.to(device), targets.to(device), sensitives.to(device)
            emb, logits = model(imgs)
            loss, _, _ = criterion(logits, emb, tgts, sens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        loss = total_loss / n
        scheduler.step()
        m = evaluate(model, val_loader, device)
        log_epoch(ep, epochs, loss, m)
        tracker.update(model, m, ep)

    print(f"done. {tracker.summary()}")


if __name__ == "__main__":
    main()
