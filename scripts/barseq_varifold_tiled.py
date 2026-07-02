# scripts/barseq_varifold_tiled.py
import os
import torch
import time
from src.config import (
    SUBSAMPLE_METHOD, KERNEL_TYPE, OPTIMIZER, SUFFIX,
    M, BASE_SIGMA, SIGMA_XY, SIGMA_Z,
    LR, MAX_ITER, HISTORY_SIZE, EPOCHS, TOL, PATIENCE,
    ADAM_LR_X, ADAM_LR_P, ADAM_EPOCHS, ADAM_TOL, ADAM_PATIENCE,
    N_GRID, strip_width, TILE_MODE, M_TOTAL,
    BPRIME_OPTIMIZE_OVERLAP, BPRIME_OVERLAP_KEEP_RATIO, BPRIME_CORE_OVERLAP,
)
from src.io.loader import load_barseq
from src.subsampling.kmeans import kmeans_subsample
from src.subsampling.random import random_subsample
from src.optim.LBFGS import optimize_lbfgs, optimize_lbfgs_joint
from src.optim.Adam import optimize_adam, optimize_adam_joint
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
del W_orig, P_norm_orig
if device.type == "cuda":
    torch.cuda.empty_cache()
regions = split_slice_grid(X_orig, P_orig, n=N_GRID,
                           strip_width=strip_width, mode=TILE_MODE)
IS_BPRIME = (TILE_MODE == "blocks_and_overlaps")


def make_varifold_fn(M_region):
    if KERNEL_TYPE == "isotropic":
        bw = BASE_SIGMA * (1.0 / max(M_region, 1)) ** (1.0 / 3.0)
        return lambda S1, S2: varifold_sp(S1, S2, bw)
    elif KERNEL_TYPE == "anisotropic":
        return lambda S1, S2: varifold_sp_anisotropic(S1, S2, SIGMA_XY, SIGMA_Z)
    raise ValueError(f"Unknown KERNEL_TYPE: '{KERNEL_TYPE}'.")


def region_measure(region):
    idx = region["idx"].to(X_orig.device)
    return X_orig[idx], P_orig[idx]


def slice_measure(S, bounds, orientation=None, seam=None, k=1):
    X, P = S
    x0, x1, y0, y1 = bounds
    mask = (X[:, 0] >= x0) & (X[:, 0] <= x1) & (X[:, 1] >= y0) & (X[:, 1] <= y1)
    if mask.any():
        return X[mask], P[mask]
    if X.shape[0] == 0 or orientation is None or seam is None:
        return X[:0], P[:0]
    axis = 0 if orientation == "vertical" else 1
    idx = torch.argsort(torch.abs(X[:, axis] - seam))[:min(k, X.shape[0])]
    return X[idx], P[idx]


def initial_measure(Xs, Ps, M_r, init=None):
    if init is not None and init[0].shape[0] >= M_r:
        X_src, P_src = init
    else:
        X_src, P_src = Xs, Ps

    if M_r >= X_src.shape[0]:
        return X_src.detach().clone().requires_grad_(True), P_src.detach().clone().requires_grad_(True)
    if SUBSAMPLE_METHOD == "kmeans":
        return kmeans_subsample(X_src.detach(), P_src.detach(), M_r, device)
    if SUBSAMPLE_METHOD == "random":
        return random_subsample(X_src.detach(), P_src.detach(), M_r)
    raise ValueError(f"Unknown SUBSAMPLE_METHOD: '{SUBSAMPLE_METHOD}'.")


def optimize_measure(S_target, M_r, init=None):
    Xs, Ps = S_target
    Ns = Xs.shape[0]
    if Ns == 0:
        return Xs[:0].detach(), Ps[:0].detach()
    M_r = max(1, min(int(M_r), Ns))

    if M_r >= Ns and init is None:
        return Xs.detach().clone(), Ps.detach().clone()

    Xh, Ph = initial_measure(Xs, Ps, M_r, init=init)
    vf = make_varifold_fn(M_r)

    if OPTIMIZER == "lbfgs":
        Xh, Ph, _, _ = optimize_lbfgs(
            S_target, Xh, Ph, vf,
            lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE,
            epochs=EPOCHS, tol=TOL, patience=PATIENCE,
        )
    elif OPTIMIZER == "adam":
        Xh, Ph, _, _ = optimize_adam(
            S_target, Xh, Ph, vf,
            lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
            epochs=ADAM_EPOCHS, tol=ADAM_TOL, patience=ADAM_PATIENCE,
        )
    else:
        raise ValueError(f"Unknown OPTIMIZER: '{OPTIMIZER}'.")
    return Xh.detach(), Ph.detach()


# ── Per-region reduction ───────────────────────────────
X_hat_list, P_hat_list, region_ids = [], [], []
t_start = time.time()

if IS_BPRIME:
    blocks = [r for r in regions if r["kind"] == "block"]
    expanded = [r for r in regions if r["kind"] == "expanded"]
    overlaps = [r for r in regions if r["kind"] == "overlap"]

    expanded_count = sum(r["n"] for r in expanded)
    M_expanded = {
        r["block_id"]: max(1, min(round(M_TOTAL * r["n"] / expanded_count), r["n"]))
        for r in expanded
    }

    expanded_hat = {}
    for region in expanded:
        M_r = M_expanded[region["block_id"]]
        Xs, Ps = region_measure(region)
        print(f"[B' expanded] {region['name']:15s} N={region['n']:6d} M={M_r}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r)
        del Xs, Ps
        expanded_hat[region["block_id"]] = (Xh.detach(), Ph.detach())

        if not BPRIME_OPTIMIZE_OVERLAP:
            X_hat_list.append(Xh.detach())
            P_hat_list.append(Ph.detach())
            region_ids.append(torch.full((Xh.shape[0],), region["block_id"], dtype=torch.int32))

    if BPRIME_OPTIMIZE_OVERLAP:
        core_bounds_key = "bounds" if BPRIME_CORE_OVERLAP else "core_bounds"
        for region in blocks:
            Xh, Ph = slice_measure(expanded_hat[region["block_id"]], region[core_bounds_key])
            print(f"[B' core]     {region['name']:15s} from expanded M={Xh.shape[0]}")
            if Xh.shape[0] == 0:
                continue
            X_hat_list.append(Xh.detach())
            P_hat_list.append(Ph.detach())
            region_ids.append(torch.full((Xh.shape[0],), region["block_id"], dtype=torch.int32))

        for rid, region in enumerate(overlaps):
            left = slice_measure(
                expanded_hat[region["u"]], region["bounds"],
                orientation=region["orientation"], seam=region["seam"], k=1
            )
            right = slice_measure(
                expanded_hat[region["v"]], region["bounds"],
                orientation=region["orientation"], seam=region["seam"], k=1
            )
            n_src = left[0].shape[0] + right[0].shape[0]
            M_r = int(n_src * BPRIME_OVERLAP_KEEP_RATIO)
            if M_r == 0:
                continue
            M_r = max(1, min(M_r, n_src))
            print(f"[B' overlap]  {region['name']:15s} M={M_r} "
                  f"left={left[0].shape[0]} right={right[0].shape[0]}")

            X_src = torch.cat([S[0].detach() for S in (left, right) if S[0].shape[0] > 0], dim=0)
            P_src = torch.cat([S[1].detach() for S in (left, right) if S[0].shape[0] > 0], dim=0)
            Xh, Ph = initial_measure(X_src, P_src, M_r)
            vf = make_varifold_fn(M_r)
            if OPTIMIZER == "lbfgs":
                Xh, Ph, _, _ = optimize_lbfgs_joint(
                    [left, right], Xh, Ph, vf,
                    lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE,
                    epochs=EPOCHS, tol=TOL, patience=PATIENCE,
                )
            elif OPTIMIZER == "adam":
                Xh, Ph, _, _ = optimize_adam_joint(
                    [left, right], Xh, Ph, vf,
                    lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
                    epochs=ADAM_EPOCHS, tol=ADAM_TOL, patience=ADAM_PATIENCE,
                )
            else:
                raise ValueError(f"Unknown OPTIMIZER: '{OPTIMIZER}'.")

            X_hat_list.append(Xh.detach())
            P_hat_list.append(Ph.detach())
            region_ids.append(torch.full((Xh.shape[0],), len(blocks) + rid, dtype=torch.int32))
else:
    counts = [r["X"].shape[0] for r in regions]
    total_n = sum(counts)
    M_alloc = [max(1, min(round(M_TOTAL * c / total_n), c)) for c in counts]

    for rid, (region, M_r) in enumerate(zip(regions, M_alloc)):
        Xs, Ps = region["X"], region["P"]
        print(f"[{rid:2d}] {region['name']:13s} kind={region['kind']:6s} "
              f"N={Xs.shape[0]:6d}  M={M_r}")

        Xh, Ph = optimize_measure((Xs, Ps), M_r)
        X_hat_list.append(Xh)
        P_hat_list.append(Ph)
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
