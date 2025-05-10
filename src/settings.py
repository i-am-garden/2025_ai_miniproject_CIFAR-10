# src/settings.py
from pathlib import Path
import torch

# ─── 실험 조건 ────────────────────────────────────────────
SETTINGS = ["baseline", "random_shuffle", "noisy20", "perturb"]

# ─── 데이터 & 배치 ───────────────────────────────────────
BATCH_TEST  = 512
BATCH_TRAIN = 128     # train.py에서 사용
NUM_EPOCHS  = 15

# ─── 경로 ────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent      # 프로젝트 루트
DATA_DIR  = ROOT / "data"
OUT_ROOT  = ROOT / "outputs"    # ← 모든 결과가 모이는 폴더
OUT_ROOT.mkdir(exist_ok=True)

# ─── 장치 ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── CIFAR-10 클래스명 ───────────────────────────────────
CLASS_NAMES = [
    "airplane", "auto", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]