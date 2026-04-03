"""
Evaluation: overall accuracy, group accuracy, fairness metrics (WGA / EqOdd).

  - Equalized Odds Difference (EqOdd) = max(|ΔTPR|, |ΔFPR|)
  - Per-sensitive-group accuracy / TPR / FPR
"""

import argparse
from collections import defaultdict
import torch
import config as cfg
from dataset import get_loader
from model import FairClassifier

# ---------- fairness metrics ----------

def compute_fairness(preds, targets, sensitives):
    """Compute fairness metrics for binary sensitive attribute."""
    groups = sorted(sensitives.unique().tolist())
    assert len(groups) == 2, f"Expected binary sensitive attr, got {groups}"
    g0, g1 = groups

    stats = {}
    for g in groups:
        mask = sensitives == g
        g_preds = preds[mask]
        g_targets = targets[mask]

        n = mask.sum().item()
        acc = (g_preds == g_targets).float().mean().item()

        tp = ((g_preds == 1) & (g_targets == 1)).sum().item()
        fp = ((g_preds == 1) & (g_targets == 0)).sum().item()
        fn = ((g_preds == 0) & (g_targets == 1)).sum().item()
        tn = ((g_preds == 0) & (g_targets == 0)).sum().item()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        stats[g] = dict(n=n, acc=acc, tpr=tpr, fpr=fpr,
                        tp=tp, fp=fp, fn=fn, tn=tn)

    eqodd = max(abs(stats[g0]["tpr"] - stats[g1]["tpr"]),
                abs(stats[g0]["fpr"] - stats[g1]["fpr"]))

    return {
        "group_stats": stats,
        "equalized_odds_diff": eqodd,
    }


def print_fairness_report(metrics):
    """Print detailed fairness report."""
    gs = metrics["group_stats"]
    sensitive_names = {0: f"Non{cfg.SENSITIVE_ATTR}", 1: cfg.SENSITIVE_ATTR}

    print("=" * 54)
    print("           Fairness Evaluation Report")
    print("=" * 54)

    header = f"{'Group':>10} {'N':>7} {'Acc':>8} {'TPR':>8} {'FPR':>8}"
    print(header)
    print("-" * 54)
    for g in sorted(gs):
        s = gs[g]
        print(f"{sensitive_names[g]:>10} {s['n']:>7d} {s['acc']:>8.2%} "
              f"{s['tpr']:>8.2%} {s['fpr']:>8.2%}")

    print("-" * 54)
    print(f"  Equalized Odds Diff : {metrics['equalized_odds_diff']:.4f}")
    print("=" * 54)
    print()
    print("  EqOdd = max(|ΔTPR|, |ΔFPR|)  → 0 is fair")
    print()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (preds, targets, sensitives) for fairness evaluation."""
    all_preds, all_targets, all_sensitives = [], [], []
    model.eval()
    for images, targets, sensitives, _ in loader:
        images = images.to(device)
        preds = model(images)[1].argmax(1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
        all_sensitives.append(sensitives)
    return torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_sensitives)


# ---------- main evaluation ----------

def compute_metrics_from_predictions(preds, targets, sensitives, groups=None):
    """Compute overall/group accuracy and fairness metrics from predictions."""
    if groups is None:
        groups = targets * 2 + sensitives

    hit = preds == targets
    total_correct = hit.sum().item()
    total_count = len(targets)

    correct, count = defaultdict(int), defaultdict(int)
    for g in range(4):
        mask = groups == g
        correct[g] = (hit & mask).sum().item()
        count[g] = mask.sum().item()

    group_acc = {g: correct[g] / max(count[g], 1) for g in range(4)}
    wg = min(group_acc, key=group_acc.get)

    fair = compute_fairness(preds, targets, sensitives)
    return {
        "overall_acc": total_correct / max(total_count, 1),
        "group_acc": group_acc,
        "worst_group_acc": group_acc[wg],
        "worst_group_id": wg,
        "eqodd": fair["equalized_odds_diff"],
        "fairness_metrics": fair,
    }

@torch.no_grad()
def evaluate(model, loader, device):
    """Return group accuracy + fairness metrics (WGA / EqOdd)."""
    preds, targets, sensitives = collect_predictions(model, loader, device)
    return compute_metrics_from_predictions(preds, targets, sensitives)


def main():
    p = argparse.ArgumentParser(description="Evaluate model: accuracy and fairness metrics")
    p.add_argument("--checkpoint", required=True, help="checkpoint path")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--report", action="store_true", help="print detailed fairness report")
    args = p.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = FairClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    m = evaluate(model, get_loader(args.split, args.bs), device)

    print(f"split={args.split}  overall={m['overall_acc']:.2%}  wga={m['worst_group_acc']:.2%}"
          f"  eqodd={m['eqodd']:.4f}")
    for g in range(4):
        tag = " <- worst" if g == m["worst_group_id"] else ""
        print(f"  {cfg.GROUP_NAMES[g]}: {m['group_acc'][g]:.2%}{tag}")

    if args.report:
        print_fairness_report(m["fairness_metrics"])


if __name__ == "__main__":
    main()
