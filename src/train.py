# src/train.py
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

from datasets import get_loader_by_setting, get_baseline_loaders
from model import SimpleCNN
from settings import (SETTINGS, NUM_EPOCHS, BATCH_TRAIN,
                      DEVICE, OUT_ROOT)


# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return running_loss / total, correct / total
# ------------------------------------------------------------------
def mixup_criterion(logits, targets):
    """
    logits : (B, C)
    targets: (y_a, y_b, lam)  with lam shape (B,)
    return  : scalar loss
    """
    y_a, y_b, lam = targets                           # each (B,)
    # CrossEntropyLoss per-sample
    ce_a = F.cross_entropy(logits, y_a, reduction='none')  # (B,)
    ce_b = F.cross_entropy(logits, y_b, reduction='none')  # (B,)
    loss = (lam * ce_a + (1.0 - lam) * ce_b).mean()        # scalar
    return loss
# ------------------------------------------------------------------
def train_one_epoch_mix(model, loader, criterion ,optimizer, device):
    model.train()
    run_loss, run_correct, total = 0.0, 0.0, 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y_a, y_b, lam = (t.to(device) for t in y)

        optimizer.zero_grad()
        logits = model(x)
        loss   = mixup_criterion(logits, (y_a, y_b, lam))
        loss.backward(); optimizer.step()

        # ── 통계 ──────────────────────────────
        run_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        run_correct += (lam * pred.eq(y_a) + (1-lam) * pred.eq(y_b)).sum().item()
        total += x.size(0)

    return run_loss / total, run_correct / total
# ------------------------------------------------------------------

# ───────── 평가 루프 (MixUp 호환) ────────────
@torch.no_grad()
def evaluate_soft(model, loader, device):
    """
    test_loader 는 항상 일반 라벨만 오므로
    기존 evaluate 와 동일하지만, tuple 이 와도 안전.
    """
    model.eval()
    correct, total = 0.0, 0

    for x, y in loader:
        x = x.to(device)

        if isinstance(y, (tuple, list)):
            y_a, y_b, lam = y
            y_a, y_b, lam = y_a.to(device), y_b.to(device), lam.to(device)
        else:
            y = y.to(device)

        logits = model(x)
        _, pred = logits.max(1)

        if isinstance(y, (tuple, list)):
            correct += (lam * pred.eq(y_a) + (1 - lam) * pred.eq(y_b)).sum().item()
            total   += x.size(0)
        else:
            correct += pred.eq(y).sum().item()
            total   += y.size(0)

    return correct / total


# ─────────────────────────────────────────────────────────
def main(args):
    out_dir = OUT_ROOT / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader
    if args.setting == "baseline":
        train_loader, test_loader = get_baseline_loaders(args.batch_size)
    else:
        train_loader = get_loader_by_setting(args.setting, args.batch_size)
        _, test_loader = get_baseline_loaders(args.batch_size)

    # Model
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 어떤 루프를 쓸지 결정
    is_mix = args.setting in {"mixup", "cutmix"}
    train_fn = train_one_epoch_mix if is_mix else train_one_epoch

    history = {"train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        _, tr_acc = train_fn(model, train_loader, criterion, optimizer, DEVICE)
        te_acc = evaluate_soft(model, test_loader, DEVICE)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(f"[{args.setting}] Epoch {epoch:02d} | "
              f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    # 저장
    torch.save(model.state_dict(), out_dir / "model.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ model saved to {out_dir / 'model.pth'}")


if __name__ == "__main__":
    # 입력 예시 : python train.py --setting baseline --epochs 15 --batch_size 128
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="baseline", choices=SETTINGS)
    p.add_argument("--epochs",  default=NUM_EPOCHS, type=int)
    p.add_argument("--batch_size", default=BATCH_TRAIN, type=int)
    args = p.parse_args()
    main(args)