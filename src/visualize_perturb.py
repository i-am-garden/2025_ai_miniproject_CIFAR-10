import torch, random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from datasets import strong_aug, rotation_aug, blur_aug, color_aug

def to_hwc(x):
    """
    Tensor (3, H, W) 또는 ndarray (3, H, W) → ndarray (H, W, 3)
    """
    if isinstance(x, torch.Tensor):
        x = x.permute(1, 2, 0).cpu().numpy()
    elif isinstance(x, np.ndarray) and x.ndim == 3:
        x = x.transpose(1, 2, 0)
    return x

# ── 글꼴 설정 (Timesnewroman으로 설정) ───────────────────────
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
})

# ── CIFAR-10 샘플 두 장 확보 (다른 클래스) ──────────────────────────
base = datasets.CIFAR10("../data", train=False, download=True,
                        transform=T.ToTensor())
idx_a, idx_b = random.sample(range(len(base)), 2)
img_a, lbl_a = base[idx_a]     # 모든 변형에 쓰일 ‘원본’
img_b, lbl_b = base[idx_b]     # mixup 두 번째 소스
cls = base.classes

# ── 변환 dict (mixup 제외) ──────────────────────────────────────────
tf_dict = {
    "baseline": T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.ToTensor()]),
    "perturb":  strong_aug,
    "rotate_random": rotation_aug,
    "blur":     blur_aug,
    "color":    color_aug,
}
lam = 0.4                                   # mixup λ
mixup_blend = lam * img_a + (1-lam) * img_b # 혼합 결과

# ── 그리드 설정 ------------------------------------------------------
cols = list(tf_dict.keys()) + ["mixup"]     # 6개 열
fig = plt.figure(figsize=(2.1*len(cols), 3.2))
gs  = fig.add_gridspec(nrows=2, ncols=len(cols), height_ratios=[1,1])

for c, name in enumerate(cols):
    ax = fig.add_subplot(gs[0, c])

    if name == "mixup":
        img = mixup_blend
        title = f"mixup λ={lam:.1f}"
    else:
        aug = tf_dict[name](img_a.permute(1,2,0).numpy())
        img = aug if isinstance(aug, np.ndarray) else aug
        title = name

    # → to_hwc 로 변환
    img_show = to_hwc(img)
    ax.imshow(img_show)
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=4)
    # # 라벨 추가
    # ax.text(0.5, -0.13, cls[lbl_a], transform=ax.transAxes,
    #         ha='center', va='top', fontsize=8)


# 두 번째 행: mixup 두 번째 소스만 채우고, 나머지는 비워 둠
ax_mix_src = fig.add_subplot(gs[1, len(cols)-1])   # 마지막 열
ax_mix_src.imshow(img_b.permute(1,2,0)); ax_mix_src.axis("off")
ax_mix_src.set_title(f"source-B\n({cls[lbl_b]})", fontsize=8, pad=4)

# 세로 중앙 정렬을 위해 빈 칸도 만들지만 axis off
for c in range(len(cols)-1):
    fig.add_subplot(gs[1, c]).axis("off")

plt.tight_layout()
plt.savefig("outputs/fig_dataset_effect.pdf", dpi=300, bbox_inches="tight")
plt.close()
