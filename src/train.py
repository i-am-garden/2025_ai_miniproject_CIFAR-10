# src/train.py
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _, pred = logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return correct / total


# ─────────────────────────────────────────────────────────
def main(args):
    out_dir = OUT_ROOT / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터
    if args.setting == "baseline":
        train_loader, test_loader = get_baseline_loaders(args.batch_size)
    else:
        train_loader = get_loader_by_setting(args.setting, args.batch_size)
        _, test_loader = get_baseline_loaders(args.batch_size)

    # 모델
    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        _, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_acc = evaluate(model, test_loader, DEVICE)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(
            f"[{args.setting}] Epoch {epoch:02d} | "
            f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}"
        )

    # 저장
    torch.save(model.state_dict(), out_dir / "model.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    # 입력 예시 : python train.py --setting baseline --epochs 15 --batch_size 128
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="baseline", choices=SETTINGS)
    p.add_argument("--epochs",  default=NUM_EPOCHS, type=int)
    p.add_argument("--batch_size", default=BATCH_TRAIN, type=int)
    args = p.parse_args()
    main(args)