# src/evaluate.py
"""
네 가지 setting(baseline, random_shuffle, noisy20, perturb)의
저장된 모델을 공통 Test set에 대해 평가하고, 다음 시각화 생성:
1) 전체 Test accuracy 막대그래프
2) 설정별 Confusion Matrix
3) Per-class Accuracy Violin plot
4) t-SNE Scatter (Baseline vs Noisy20 비교용)
"""

from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from model import SimpleCNN
from settings import (SETTINGS, BATCH_TEST as BATCH,
                      DEVICE, OUT_ROOT, DATA_DIR, CLASS_NAMES)
# ───────── DataLoader ─────────────────────────────────────
@torch.no_grad()
def get_test_loader():
    tf = transforms.ToTensor()
    test_set = datasets.CIFAR10(ROOT / "data", train=False,
                                download=True, transform=tf)
    return torch.utils.data.DataLoader(
        test_set, batch_size=BATCH, shuffle=False, num_workers=2
    ), test_set

# ───────── 모델 평가 ──────────────────────────────────────
@torch.no_grad()
def eval_model(weight_path, loader):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    total, correct = 0, 0
    all_true, all_pred = [], []
    feats = []                       # t-SNE용 feature 저장
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        _, pred = logits.max(1)

        # feature: GAP 직전 벡터 (B,128)
        f = model.features(x).view(x.size(0), -1).cpu()
        feats.append(f)

        total += y.size(0)
        correct += pred.eq(y).sum().item()
        all_true.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())

    accuracy = correct / total
    cm = confusion_matrix(all_true, all_pred)
    feats = torch.cat(feats, dim=0)       # (N_test, 128)
    return accuracy, cm, feats.numpy(), np.array(all_true)

# ───────── 메인 루프 ──────────────────────────────────────
def main():
    test_loader, test_set = get_test_loader()

    acc_dict, cm_dict, feat_dict, label_arr = {}, {}, {}, None
    for s in SETTINGS:
        weight = OUT_ROOT / s / "model.pth"
        if not weight.exists():
            print(f"[skip] {s:15s}  (model file not found)")
            continue
        acc, cm, feats, y_true = eval_model(weight, test_loader)
        acc_dict[s], cm_dict[s], feat_dict[s] = acc, cm, feats
        label_arr = y_true                    # 동일하므로 한 번만 저장
        print(f"[{s:15s}] accuracy = {acc:.4f}")

    # ── (1) Accuracy bar ─────────────────────────────────
    plt.figure(figsize=(6, 4))
    names = list(acc_dict.keys())
    vals = [acc_dict[k] for k in names]
    sns.barplot(x=names, y=vals, palette="deep")
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("CIFAR-10 Accuracy by Setting")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "accuracy_by_setting.png", dpi=150)
    plt.close()

    # ── (2) Confusion Matrices ───────────────────────────
    for s, cm in cm_dict.items():
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix – {s}")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / f"cm_{s}.png", dpi=150)
        plt.close()

    # ── (3) Per-class Accuracy Violin ────────────────────
    per_cls = {s: cm.diagonal() / cm.sum(1) for s, cm in cm_dict.items()}
    # DataFrame 형태로 변환 (class × setting)
    import pandas as pd
    df = pd.DataFrame(per_cls, index=CLASS_NAMES).reset_index().melt(
        id_vars="index", var_name="setting", value_name="accuracy"
    )
    plt.figure(figsize=(8, 4))
    sns.violinplot(x="index", y="accuracy", hue="setting",
                   data=df, split=True, inner="quart", palette="muted")
    plt.xticks(rotation=45)
    plt.ylabel("Per-class Accuracy")
    plt.xlabel("Class")
    plt.title("Per-class Accuracy Distribution")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "per_class_accuracy_violin.png", dpi=150)
    plt.close()

    # ── (4) t-SNE (Baseline vs Noisy20) ───────────────────
    if "baseline" in feat_dict and "noisy20" in feat_dict:
        # 2000개 샘플만 랜덤 추출 → 시각 복잡도 줄임
        idx = random.sample(range(len(label_arr)), k=2000)
        feats_base = feat_dict["baseline"][idx]
        feats_noisy = feat_dict["noisy20"][idx]
        labels_sub = label_arr[idx]

        X = np.vstack([feats_base, feats_noisy])
        y = np.hstack([labels_sub, labels_sub])
        domain = np.array([0] * len(idx) + [1] * len(idx))  # 0=base,1=noisy

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb = tsne.fit_transform(X)

        plt.figure(figsize=(6, 5))
        # 점 색상: 클래스 / 모양: 도메인
        for cls in range(10):
            for d, marker in zip([0, 1], ['o', 's']):
                mask = (y == cls) & (domain == d)
                plt.scatter(emb[mask, 0], emb[mask, 1],
                            s=14, alpha=0.6,
                            label=f"{CLASS_NAMES[cls]} – {'base' if d==0 else 'noisy'}",
                            marker=marker)
        plt.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.05, 1))
        plt.title("t-SNE of Penultimate Features\n(Baseline vs Noisy20)")
        plt.tight_layout()
        plt.savefig(OUT_ROOT / "tsne_base_vs_noisy20.png", dpi=150)
        plt.close()

    print("✓ All plots saved to", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
