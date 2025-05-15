# src/stat_significant.py  (v2: pair-wise mixup comparison)
"""
mixup 유무(paired) 통계적 유의성 검정  ────────────────────────────────
각 짝(base ↔ base_mixup)의 Top-k 정확도 차이를
    • McNemar χ² test  • Bootstrap 95 % CI
로 평가해 표 + JSON 출력.

사용 예)
    python stat_significant.py          # Top-1 (기본), 모든 짝
    python stat_significant.py --k 3    # Top-3
"""

from pathlib import Path
import argparse, json
import numpy as np, torch
from tqdm import tqdm
from statsmodels.stats.contingency_tables import mcnemar
from torchvision import datasets, transforms
from model import SimpleCNN
from settings import (SETTINGS, DEVICE, OUT_ROOT, DATA_DIR)

# ───────── 데이터 로더 ─────────────────────────────────────────────
def get_test_loader(batch=128):
    tf = transforms.ToTensor()
    test = datasets.CIFAR10(DATA_DIR, train=False,
                            download=True, transform=tf)
    return torch.utils.data.DataLoader(test, batch_size=batch,
                                       shuffle=False, num_workers=2)

@torch.no_grad()
def get_correct_vec(weight_path, loader, k=1):
    """weight → 1/0 correct 벡터 (길이=테스트샘플수)"""
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    ok = []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        if k == 1:
            ok.append(logits.argmax(1).eq(y.to(DEVICE)).cpu())
        else:
            topk = logits.topk(k, 1, True, True)[1]
            ok.append(topk.eq(y.view(-1,1).to(DEVICE)).any(1).cpu())
    return torch.cat(ok).numpy().astype(int)

# ───────── 통계 함수 ───────────────────────────────────────────────
def mcnemar_pair(b_ok, m_ok):
    n01 = int(((b_ok == 1) & (m_ok == 0)).sum())
    n10 = int(((b_ok == 0) & (m_ok == 1)).sum())
    res = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True)
    return res.statistic, res.pvalue, n01, n10

def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng, n = np.random.default_rng(seed), diff.size
    boots = [diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])

# ───────── 메인 ───────────────────────────────────────────────────
def main(args):
    loader = get_test_loader()
    # 1) 짝 찾기  (baseline↔mixup, *_mixup↔* )
    pairs = []
    mixups = {s for s in SETTINGS if s.endswith('_mixup') or s == 'mixup'}
    for s in SETTINGS:
        if s == 'mixup':          # baseline 전용 짝
            pairs.append(('baseline', 'mixup'))
        elif s.endswith('_mixup'):
            base = s[:-6]         # '_mixup' 제거
            if base in SETTINGS:
                pairs.append((base, s))
    # 사용자가 짝을 직접 주면 그 목록만
    if args.pairs:
        pairs = [tuple(p.split(',')) for p in args.pairs]

    results = {}
    for base, mix in pairs:
        w_base = OUT_ROOT / base / 'model.pth'
        w_mix  = OUT_ROOT / mix  / 'model.pth'
        if not (w_base.exists() and w_mix.exists()):
            print(f"[skip] {base}↔{mix}  (model.pth 없음)")
            continue

        b_ok = get_correct_vec(w_base, loader, args.k)
        m_ok = get_correct_vec(w_mix,  loader, args.k)

        chi2, p, n01, n10 = mcnemar_pair(b_ok, m_ok)
        diff = m_ok - b_ok
        ci_lo, ci_hi = bootstrap_ci(diff, args.n_boot)
        key = f"{base}↔{mix}"
        results[key] = dict(
            acc_base=float(b_ok.mean()),
            acc_mix=float(m_ok.mean()),
            delta=float(diff.mean()),
            ci=[float(ci_lo), float(ci_hi)],
            n01=n01, n10=n10,
            chi2=float(chi2), pvalue=float(p)
        )

    # ---- 출력 ----
    print(f"\nTop-{args.k} Accuracy  mixup vs non-mixup")
    print("-"*86)
    print(f"{'pair':25s}  acc_base  acc_mix   Δacc   95% CI          χ²    p")
    for k, m in results.items():
        print(f"{k:25s}  {m['acc_base']:.4f}  {m['acc_mix']:.4f}  "
              f"{m['delta']:+.4f}  [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]  "
              f"{m['chi2']:6.2f}  {m['pvalue']:.4f}")
    print("-"*86)

    out = OUT_ROOT / f"stat_sig_pairs_top{args.k}.json"
    out.write_text(json.dumps(results, indent=2))
    print("✓ 결과 저장:", out.resolve())

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, help="Top-k (1|3)")
    ap.add_argument("--n_boot", type=int, default=10000,
                    help="Bootstrap 반복 횟수")
    ap.add_argument("--pairs", nargs="*",
                    help="직접 지정할 짝: 'base,mixup' 형식 (공백 구분)")
    main(ap.parse_args())

# 사용방법 :
# python src/stat_significant.py
# python src/stat_significant.py --k 3 --n_boot 10000
# python src/stat_significant.py --pairs baseline,mixup noisy20,mixup
# python src/stat_significant.py --pairs baseline,mixup noisy20,mixup random_shuffle,mixup