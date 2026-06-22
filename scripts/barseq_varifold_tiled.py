# scripts/barseq_varifold_tiled.py
import os
import torch
import time
from src.config import (
    SUBSAMPLE_METHOD, KERNEL_TYPE, OPTIMIZER, SUFFIX,
    M, BASE_SIGMA, SIGMA_XY, SIGMA_Z,
    LR, MAX_ITER, HISTORY_SIZE, EPOCHS, TOL, PATIENCE,
    ADAM_LR_X, ADAM_LR_P, ADAM_EPOCHS, ADAM_TOL, ADAM_PATIENCE,
    N_GRID, strip_width, TILE_MODE, M_TOTAL
)
from src.io.loader import load_barseq
from src.subsampling.kmeans import kmeans_subsample
from src.subsampling.random import random_subsample
from src.optim.LBFGS import optimize_lbfgs
from src.optim.Adam import optimize_adam
from src.losses.varifold import varifold_sp, varifold_sp_anisotropic
from src.io.vtk_export import export_orig_vtp, export_hat_vtp
from src.preprocessing.tiling import split_slice_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Tiling: {N_GRID}x{N_GRID}, strip_width={strip_width}, mode={TILE_MODE}")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "npz_slices", "slice_10.npz")
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "MB35_BL2_L20_11.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load + tile ────────────────────────────────────────
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(DATA_FILE, device)
regions = split_slice_grid(X_orig, P_orig, n=N_GRID,
                           strip_width=strip_width, mode=TILE_MODE)

# ── Allocate M proportionally to each region's point count ─────────
counts  = [r["X"].shape[0] for r in regions]
total_n = sum(counts)
M_alloc = [max(1, min(round(M_TOTAL * c / total_n), c)) for c in counts]


def make_varifold_fn(M_region):
    if KERNEL_TYPE == "isotropic":
        bw = BASE_SIGMA * (1.0 / max(M_region, 1)) ** (1.0 / 3.0)
        return lambda S1, S2: varifold_sp(S1, S2, bw)
    elif KERNEL_TYPE == "anisotropic":
        return lambda S1, S2: varifold_sp_anisotropic(S1, S2, SIGMA_XY, SIGMA_Z)
    raise ValueError(f"Unknown KERNEL_TYPE: '{KERNEL_TYPE}'.")


# ── Per-region reduction ───────────────────────────────
X_hat_list, P_hat_list, region_ids = [], [], []
t_start = time.time()

for rid, (region, M_r) in enumerate(zip(regions, M_alloc)):
    Xs, Ps = region["X"], region["P"]
    Ns = Xs.shape[0]
    print(f"[{rid:2d}] {region['name']:13s} kind={region['kind']:6s} "
          f"N={Ns:6d}  M={M_r}")

    if Ns == 0:
        continue

    # too few points to compress -> keep them as-is
    if M_r >= Ns:
        X_hat_list.append(Xs.detach().clone())
        P_hat_list.append(Ps.detach().clone())
        region_ids.append(torch.full((Ns,), rid, dtype=torch.int32))
        continue

    if SUBSAMPLE_METHOD == "kmeans":
        Xh, Ph = kmeans_subsample(Xs, Ps, M_r, device)
    elif SUBSAMPLE_METHOD == "random":
        Xh, Ph = random_subsample(Xs, Ps, M_r)
    else:
        raise ValueError(f"Unknown SUBSAMPLE_METHOD: '{SUBSAMPLE_METHOD}'.")

    vf = make_varifold_fn(M_r)
    S_region = (Xs, Ps)   # IMPORTANT: each region is matched to its OWN original

    if OPTIMIZER == "lbfgs":
        Xh, Ph, _, _ = optimize_lbfgs(
            S_region, Xh, Ph, vf,
            lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE,
            epochs=EPOCHS, tol=TOL, patience=PATIENCE,
        )
    elif OPTIMIZER == "adam":
        Xh, Ph, _, _ = optimize_adam(
            S_region, Xh, Ph, vf,
            lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
            epochs=ADAM_EPOCHS, tol=ADAM_TOL, patience=ADAM_PATIENCE,
        )
    else:
        raise ValueError(f"Unknown OPTIMIZER: '{OPTIMIZER}'.")

    X_hat_list.append(Xh.detach())
    P_hat_list.append(Ph.detach())
    region_ids.append(torch.full((Xh.shape[0],), rid, dtype=torch.int32))

total_time = time.time() - t_start
print(f"Total reduction time      : {total_time:.2f} s")
# ── Merge ──────────────────────────────────────────────
X_hat_all = torch.cat(X_hat_list, dim=0)
P_hat_all = torch.cat(P_hat_list, dim=0)
print(f"Total representative points after tiling: {X_hat_all.shape[0]} "
      f"(orig {X_orig.shape[0]})")

# ── Export ─────────────────────────────────────────────
out_suffix = f"{SUFFIX}_tiled{N_GRID}x{N_GRID}"
export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
export_hat_vtp(X_hat_all, P_hat_all, X_min, X_max, OUTPUT_DIR, suffix=out_suffix)
'''
S_full    = (X_orig, P_orig)
S_hat_all = (X_hat_all, P_hat_all)
vf_global = make_varifold_fn(M_TOTAL)

with torch.no_grad():
    term0   = vf_global(S_full,    S_full)     # <S, S>
    term1   = vf_global(S_hat_all, S_hat_all)  # <Ŝ, Ŝ>
    term2   = vf_global(S_full,    S_hat_all)  # <S, Ŝ>
    dist_sq = (term0 + term1 - 2 * term2).item()

dist = dist_sq ** 0.5 if dist_sq > 0 else float("nan")

print("──────────────────────────────────────────")
print(f"Merged varifold distance² : {dist_sq:.6e}")
print(f"Merged varifold distance  : {dist:.6e}")
print(f"Total reduction time      : {total_time:.2f} s")
print("──────────────────────────────────────────")
'''
