# src/evaluate.py
"""
평가/시각화
1) Top-1 Accuracy 막대그래프
2) Top-3 Accuracy 막대그래프
3) Δ-Accuracy Heatmap (setting × class)
4) Confusion Matrix 각 setting
5) Aggregated Mis-classification Δ-Heatmap
"""

from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

from model import SimpleCNN
from settings import (SETTINGS, BATCH_TEST as BATCH,
                      DEVICE, OUT_ROOT, DATA_DIR, CLASS_NAMES)

# ───────── 헬퍼 ───────────────────────────────────────────
@torch.no_grad()
def get_test_loader():
    tf = transforms.ToTensor()
    test = datasets.CIFAR10(DATA_DIR, train=False,
                            download=True, transform=tf)
    loader = torch.utils.data.DataLoader(test, batch_size=BATCH,
                                         shuffle=False, num_workers=2)
    return loader

def accuracy_topk(logits, targets, k=3):
    topk = logits.topk(k, 1, True, True)[1]       # (B,k)
    return topk.eq(targets.view(-1, 1).expand_as(topk)).any(1)

# ───────── 모델 평가 ──────────────────────────────────────
@torch.no_grad()
def eval_model(weight_path, loader):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    tot = hit1 = hit3 = 0
    cls_tot  = np.zeros(10, int)
    cls_hit1 = np.zeros(10, int)

    y_true_all, y_pred_all = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)

        pred1   = logits.argmax(1)
        top3_ok = accuracy_topk(logits, y, k=3)

        # 전체
        tot  += y.size(0)
        hit1 += pred1.eq(y).sum().item()
        hit3 += top3_ok.sum().item()

        # per-class
        for c in range(10):
            mask = (y == c)
            if mask.any():
                cls_tot[c]  += mask.sum().item()
                cls_hit1[c] += pred1[mask].eq(c).sum().item()

        y_true_all.extend(y.cpu().numpy())
        y_pred_all.extend(pred1.cpu().numpy())

    acc1 = hit1 / tot
    acc3 = hit3 / tot

    per_cls_acc1 = cls_hit1 / cls_tot         # (10,)
    cm = confusion_matrix(y_true_all, y_pred_all)

    return acc1, acc3, per_cls_acc1, cm

# ───────── 메인 ───────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loader = get_test_loader()

    meta, cm_dict = {}, {}
    for s in SETTINGS:
        w = OUT_ROOT / s / "model.pth"
        if not w.exists():
            print(f"[skip] {s:15s}  (model file not found)")
            continue
        acc1, acc3, per_cls, cm = eval_model(w, loader)
        err1, err3 = 1-acc1, 1-acc3
        meta[s]          = dict(acc1=float(acc1),
                                acc3=float(acc3),
                                err1=float(err1),
                                err3=float(err3),
                                per_cls=per_cls.tolist())
        cm_dict[s] = cm
        print(f"[{s:15s}]  top1={acc1:.4f}  top3={acc3:.4f}  "
              f"err1={err1:.4f}  err3={err3:.4f}")

    names = list(meta.keys())
    # ---------- (1) Top-1 Accuracy bar -------------------
    plt.figure(figsize=(6,4))
    sns.barplot(x=names, y=[meta[k]['acc1'] for k in names],
                hue=names, legend=False, palette="deep")
    plt.ylim(0,1); plt.ylabel("Top-1 Acc"); plt.title("Top-1 Accuracy")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"acc_top1.png", dpi=150); plt.close()

    # ---------- (2) Top-3 Accuracy bar -------------------
    plt.figure(figsize=(6,4))
    sns.barplot(x=names, y=[meta[k]['acc3'] for k in names],
                hue=names, legend=False, palette="crest")
    plt.ylim(0,1); plt.ylabel("Top-3 Acc"); plt.title("Top-3 Accuracy")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"acc_top3.png", dpi=150); plt.close()

    # ---------- (3) Δ-Accuracy Heatmap -------------------
    base = np.asarray(meta["baseline"]["per_cls"])
    delta_rows, ylbl = [], []
    for s in names:
        if s=="baseline": continue
        delta_rows.append(np.asarray(meta[s]["per_cls"]) - base)
        ylbl.append(s)
    delta_mat = np.vstack(delta_rows) if delta_rows else np.empty((0,10))
    plt.figure(figsize=(10,1.5+0.4*len(ylbl)))
    sns.heatmap(delta_mat, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                xticklabels=CLASS_NAMES, yticklabels=ylbl)
    plt.title("Δ Accuracy (setting – baseline)")
    plt.xlabel("Class"); plt.ylabel("Setting")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"delta_acc_heatmap.png", dpi=150); plt.close()

    # ---------- (4) Confusion Matrices -------------------
    for s, cm in cm_dict.items():
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"Confusion Matrix – {s}")
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(OUT_ROOT/f"cm_{s}.png", dpi=150); plt.close()

    # ---------- (5) Aggregated Mis-Δ Heatmap -------------
    base_cm = cm_dict["baseline"].astype(int)
    mis_delta = np.zeros_like(base_cm)
    for s, cm in cm_dict.items():
        if s=="baseline": continue
        diff = cm.astype(int) - base_cm
        diff[np.diag_indices_from(diff)] = 0      # diagonal 제거
        mis_delta += diff
    plt.figure(figsize=(7,6))
    sns.heatmap(mis_delta, mask=(mis_delta==0), cmap="RdBu_r", center=0,
                annot=True, fmt="+d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws=dict(label="Δ count (aggregated)"))
    plt.title("Aggregated Mis-classification Δ  (all settings vs baseline)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"mis_delta_heatmap.png", dpi=150); plt.close()
    # (6) Top-k Error bar ---------------------------------
    plt.figure(figsize=(6,4))
    sns.barplot(x=names, y=[meta[k]['err1'] for k in names],
                hue=names, legend=False, palette="rocket")
    plt.ylim(0,1); plt.ylabel("Top-1 Error"); plt.title("Top-1 Error")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"err_top1.png", dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    sns.barplot(x=names, y=[meta[k]['err3'] for k in names],
                hue=names, legend=False, palette="mako_r")
    plt.ylim(0,1); plt.ylabel("Top-3 Error"); plt.title("Top-3 Error")
    plt.tight_layout(); plt.savefig(OUT_ROOT/"err_top3.png", dpi=150); plt.close()

    # ---------- JSON 저장 --------------------------------
    with open(OUT_ROOT/"metrics_topk.json","w") as fp:
        json.dump(meta, fp, indent=2)
    print("✓ plots & metrics saved to", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
