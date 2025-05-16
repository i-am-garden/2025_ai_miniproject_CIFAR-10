import torch, random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
from datasets import strong_aug, rotation_aug, blur_aug, color_aug

def to_hwc(x):
    if isinstance(x, torch.Tensor):
        x = x.permute(1, 2, 0).cpu().numpy()
    elif isinstance(x, np.ndarray) and x.ndim == 3:
        x = x.transpose(1, 2, 0)
    return x

# ── 폰트 설정 ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,              # 전반적 폰트 크기 ↑
    "axes.titlesize": 14,         # 축 제목 크기 ↑
    "figure.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ── CIFAR-10 샘플 준비 ────────────────────────────────────
base = datasets.CIFAR10("../data", train=False, download=True,
                        transform=T.ToTensor())
idx_a, idx_b = random.sample(range(len(base)), 2)
img_a, lbl_a = base[idx_a]
img_b, lbl_b = base[idx_b]
cls = base.classes

tf_dict = {
    "baseline": T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.ToTensor()]),
    "perturb":  strong_aug,
    "rotate_random": rotation_aug,
    "blur":     blur_aug,
    "color":    color_aug,
}
lam = 0.4
mixup_blend = lam * img_a + (1-lam) * img_b

# ── 그리드 & Figure 생성 ───────────────────────────────────────
cols = list(tf_dict.keys()) + ["mixup"]
fig = plt.figure(
    figsize=(2.1*len(cols), 3.50),  # 높이를 약간 늘려줍니다
    constrained_layout=False       # subplots_adjust를 쓸 거라 끕니다
)
gs = fig.add_gridspec(
    nrows=2, ncols=len(cols),
    height_ratios=[1, 0.7],        # 상단 그림에 비해 하단 영역 조금 더 확보
    wspace=0.0, hspace=0.2         # 행 간격(hspace)을 약간 줘서 겹침 방지
)

# ── 상단 변형 시각화 ─────────────────────────────────────────
for c, name in enumerate(cols):
    ax = fig.add_subplot(gs[0, c])
    if name == "mixup":
        img = mixup_blend
        title = f"mixup λ={lam:.1f}"
    else:
        aug = tf_dict[name](img_a.permute(1,2,0).numpy())
        img = aug if isinstance(aug, np.ndarray) else aug
        title = name

    ax.imshow(to_hwc(img))
    ax.axis("off")
    ax.set_title(title, pad=6)   # pad 조금 늘려서 제목↔그림 간 여백 확보

# ── 하단 mixup 원본 시각화 ────────────────────────────────────
# 나머지는 빈 축으로
for c in range(len(cols)-1):
    ax = fig.add_subplot(gs[1, c])
    ax.axis("off")

# 마지막 열에만 원본 B 이미지
ax_b = fig.add_subplot(gs[1, len(cols)-1])
ax_b.imshow(to_hwc(img_b))
ax_b.axis("off")
ax_b.set_title(f"source-B\n({cls[lbl_b]})",
               fontsize=10,  # 하단 제목은 약간 작게
               pad=4)

# ── 전체 여백 제거 ─────────────────────────────────────────
fig.subplots_adjust(
    left=0, right=1, top=1, bottom=0,
    wspace=0.0, hspace=0.1
)

plt.savefig(
    "outputs/fig_dataset_effect.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)
plt.close()
