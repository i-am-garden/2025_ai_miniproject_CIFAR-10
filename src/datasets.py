# src/datasets.py
from typing import Tuple
import torch
import numpy as np
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────
# 1) Baseline DataLoader (표준 증강) ───────────────────────
# ─────────────────────────────────────────────────────────
def get_baseline_loaders(batch_size: int = 128):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.ToTensor()

    train_set = datasets.CIFAR10(
        root="../data", train=True, download=True, transform=tf_train
    )
    test_set = datasets.CIFAR10(
        root="../data", train=False, download=True, transform=tf_test
    )

    return (
        torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        ),
        torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        ),
    )


# ─────────────────────────────────────────────────────────
# 2) 커스텀 데이터셋 (라벨／데이터 변형) ────────────────────
# ─────────────────────────────────────────────────────────
class RandomLabelDataset(torch.utils.data.Dataset):
    """학습용 라벨을 전부 무작위로 섞은 버전"""

    def __init__(self, base_set: datasets.CIFAR10):
        self.data = base_set.data
        self.targets = np.random.permutation(base_set.targets)
        self.transform = base_set.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx], int(self.targets[idx])
        return self.transform(img), lbl


class NoisyLabelDataset(torch.utils.data.Dataset):
    """일부 라벨(기본 20%)을 다른 클래스로 치환"""

    def __init__(
        self, base_set: datasets.CIFAR10, noise_ratio: float = 0.2, n_classes: int = 10
    ):
        self.data = base_set.data
        self.targets = np.array(base_set.targets)
        n_noisy = int(noise_ratio * len(self.targets))
        idx = np.random.choice(len(self.targets), n_noisy, replace=False)
        self.targets[idx] = np.random.randint(0, n_classes, size=n_noisy)
        self.transform = base_set.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx], int(self.targets[idx])
        return self.transform(img), lbl

class MixupDataset(torch.utils.data.Dataset):
    """β(α,α) 분포에서 λ를 뽑아 두 샘플을 선형 혼합"""
    #lambda값은 0~1사이의 값으로 두 샘플을 섞는 비율을 나타냄
    #α값이 클수록 λ는 0.5에 가까워지고, α값이 작을수록 λ는 0.0 또는 1.0에 가까워짐
    #구체적인 수식은 λ = np.random.beta(α, α)로 표현됨

    def __init__(self, base_set: datasets.CIFAR10, alpha: float = 0.4):
        self.data = base_set.data
        self.targets = np.array(base_set.targets)
        self.transform = base_set.transform
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def _rand_pair(self, idx):
        j = np.random.randint(0, len(self.data))
        return j

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
        img1, lbl1 = self.data[idx], int(self.targets[idx])
        j = self._rand_pair(idx)
        img2, lbl2 = self.data[j], int(self.targets[j])

        lam = np.random.beta(self.alpha, self.alpha)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img, (lbl1, lbl2, lam)


# 강한 이미지 변형(입력 섭동)
strong_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(0.9, 0.9, 0.9, 0.3),
        transforms.ToTensor(),
    ]
)

blur_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ]
)

color_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.ColorJitter(0.9, 0.9, 0.9, 0.3),
        transforms.ToTensor(),
    ]
)

# datasets.py (맨 위 import 근처에 추가해도 되고 파일 아래쪽에 둬도 됩니다)
rotation_aug = transforms.Compose([
    transforms.RandomRotation(
        degrees=180,                     # (-180°, +180°) 범위
        interpolation=transforms.InterpolationMode.BILINEAR,
        expand=False,                   # True로 두면 모서리 잘림 방지 대신 padding 생김
        fill=0                          # 잘린 영역은 검정(=0)으로 채움
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────────────────
# 3) 설정 이름 → DataLoader 매핑 함수 ─────────────────────
# ─────────────────────────────────────────────────────────
def get_loader_by_setting(setting: str, batch_size: int = 128):
    """setting ∈ {'baseline','random_shuffle','noisy20','perturb'}"""
    # 공통: 원본 train 세트 (라벨은 아직 수정 안 함)
    base_train = datasets.CIFAR10(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    )

    if setting == "baseline":
        base_train.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )
        train_set = base_train
    elif setting == "random_shuffle":
        train_set = RandomLabelDataset(base_train)
    elif setting == "noisy20":
        train_set = NoisyLabelDataset(base_train, noise_ratio=0.2)
    elif setting == "perturb":
        base_train.transform = strong_aug
        train_set = base_train
    elif setting == "mixup":
        train_set = MixupDataset(base_train, alpha=0.4)
    elif setting == "rotaterandom":
        base_train.transform = rotation_aug
        train_set = base_train
    elif setting == "blur":
        base_train.transform = blur_aug
        train_set = base_train
    elif setting == "color":
        base_train.transform = color_aug
        train_set = base_train
    elif setting == "noisy20_mixup":
        train_set = MixupDataset(
            NoisyLabelDataset(base_train, noise_ratio=0.2), alpha=0.4
        )
    elif setting == "blur_mixup":
        train_set = MixupDataset(
            datasets.CIFAR10(
                "../data", train=True, download=True, transform=blur_aug
            ),
            alpha=0.4,
        )
    elif setting == "color_mixup":
        train_set = MixupDataset(
            datasets.CIFAR10(
                "../data", train=True, download=True, transform=color_aug
            ),
            alpha=0.4,
        )
    elif setting == "perturb_mixup":
        train_set = MixupDataset(
            datasets.CIFAR10(
                "../data", train=True, download=True, transform=strong_aug
            ),
            alpha=0.4,
        )
    elif setting == "rotaterandom_mixup":
        train_set = MixupDataset(
            datasets.CIFAR10(
                "../data", train=True, download=True, transform=rotation_aug
            ),
            alpha=0.4,
        )
    else:
        raise ValueError(f"Unknown setting: {setting}")

    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
