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
    BPRIME_OPTIMIZE_OVERLAP, BPRIME_OVERLAP_KEEP_RATIO,
    STRIP_MOVE_P, Input
)
from src.io.loader import load_barseq
from src.subsampling.kmeans import kmeans_subsample
from src.subsampling.random import random_subsample
from src.optim.LBFGS import optimize_lbfgs, optimize_lbfgs_joint
from src.optim.Adam import optimize_adam, optimize_adam_joint
from src.losses.varifold import varifold_sp, varifold_sp_anisotropic
from src.io.vtk_export import export_orig_vtp, export_hat_vtp
from src.preprocessing.tiling import split_slice_grid, strip_geometry


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Tiling: {N_GRID}x{N_GRID}, strip_width={strip_width}, mode={TILE_MODE}")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "npz_slices", "slice_10.npz")
VTK_FILE = os.path.join(BASE_DIR, "data", "BARSeq", Input)
#DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", ".npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load + tile ────────────────────────────────────────
# Keep the (large) original cloud resident on CPU — P_orig is (N, 39) and on its own
# nearly fills a small GPU. Only per-region / per-strip subsets are moved to `device`.
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(VTK_FILE, torch.device("cpu"))
del W_orig, P_norm_orig
if device.type == "cuda":
    torch.cuda.empty_cache()
regions = split_slice_grid(X_orig, P_orig, n=N_GRID,
                           strip_width=strip_width, mode=TILE_MODE)
IS_BPRIME = (TILE_MODE == "blocks_and_overlaps")
IS_BSTRIP = (TILE_MODE == "blocks_and_strips")


def make_varifold_fn(Mr):
    if KERNEL_TYPE == "isotropic":
        #bw = BASE_SIGMA * (1.0 / max(M_TOTAL, 1)) ** (1.0 / 3.0)
        bw = BASE_SIGMA
        return lambda S1, S2: varifold_sp(S1, S2, bw)
    elif KERNEL_TYPE == "anisotropic":
        return lambda S1, S2: varifold_sp_anisotropic(S1, S2, SIGMA_XY, SIGMA_Z)
    raise ValueError(f"Unknown KERNEL_TYPE: '{KERNEL_TYPE}'.")


def region_measure(region):
    idx = region["idx"]
    return X_orig[idx].to(device), P_orig[idx].to(device)


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


def refine_strip(S_target, X_mov, P_mov, alpha):
    """
    Smooth blend between the block optimum and a free strip re-optimization.

    X_mov/P_mov (= mu_hat) are the optimized-block points inside the seam band. They are
    re-optimized *freely* against the original data S_target in the same band to get
    mu_tilde (edges free too — continuity is handled by the blend, not by freezing), then
    interpolated per point:

        X#  =  X_hat + rho * (X_tilde - X_hat)

    with rho = `alpha`, the hat weight: rho=0 at the strip edges keeps the block optimum
    (stable interior of the block), rho=1 at the seam centre adopts the strip optimum
    (fixes the block-to-block junction). Features are blended the same way when
    STRIP_MOVE_P, otherwise kept at P_hat.
    """
    if X_mov.shape[0] == 0 or S_target[0].shape[0] == 0:
        return X_mov.detach(), P_mov.detach()

    X_hat = X_mov.detach().clone()          # mu_hat: block optimum (blend baseline)
    P_hat = P_mov.detach().clone()

    # mu_tilde: free re-optimization of the same points (same count/order → 1:1 blend).
    X_t = X_hat.clone().requires_grad_(True)
    P_t = P_hat.clone().requires_grad_(STRIP_MOVE_P)
    vf = make_varifold_fn(X_mov.shape[0])

    if OPTIMIZER == "lbfgs":
        X_t, P_t, _, _ = optimize_lbfgs(
            S_target, X_t, P_t, vf,
            lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE,
            epochs=EPOCHS, tol=TOL, patience=PATIENCE,
        )
    elif OPTIMIZER == "adam":
        X_t, P_t, _, _ = optimize_adam(
            S_target, X_t, P_t, vf,
            lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
            epochs=ADAM_EPOCHS, tol=ADAM_TOL, patience=ADAM_PATIENCE,
        )
    else:
        raise ValueError(f"Unknown OPTIMIZER: '{OPTIMIZER}'.")

    rho = alpha.detach().view(-1, 1)                       # (M,1), in [0,1]
    X_sharp = X_hat + rho * (X_t.detach() - X_hat)         # X# = X_hat + rho*(X_tilde-X_hat)
    if STRIP_MOVE_P:
        P_sharp = (P_hat + rho * (P_t.detach() - P_hat)).clamp(min=0.0)
    else:
        P_sharp = P_hat
    return X_sharp.detach(), P_sharp.detach()


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
        for region in blocks:
            Xh, Ph = slice_measure(expanded_hat[region["block_id"]], region["bounds"])
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
elif IS_BSTRIP:
    # Plan B (revised): optimize the block partition first, then carve thin seam bands out
    # of that optimized cloud and re-optimize them against the ORIGINAL data with the block
    # junctions frozen (hat weight). Blocks-minus-seams and refined seams are complementary,
    # so the merge is non-overlapping and point-count-preserving.
    geom = strip_geometry(N_GRID, strip_width=strip_width)
    h = geom["h"]
    # Lines live on two devices: GPU copy classifies the (small) optimized block cloud;
    # CPU copy classifies the (large) original cloud without moving it to the GPU.
    vlines_g = torch.tensor(geom["vlines"], device=device)
    hlines_g = torch.tensor(geom["hlines"], device=device)
    vlines_c = vlines_g.cpu()
    hlines_c = hlines_g.cpu()

    def nearest_line(coord, lines):
        # Uniform interior grid → nearest line by rounding, no (N, L) temporary.
        step = (lines[-1] - lines[0]) / (lines.shape[0] - 1) if lines.shape[0] > 1 else 1.0
        k = torch.round((coord - lines[0]) / step).clamp_(0, lines.shape[0] - 1).long()
        return (coord - lines[k]).abs(), k

    # 1) optimize each block (full partition), same budget split as blocks_only.
    counts = [r["n"] for r in regions]
    total_n = sum(counts)
    M_alloc = [max(1, min(round(M_TOTAL * c / total_n), c)) for c in counts]
    Xb_list, Pb_list = [], []
    for region, M_r in zip(regions, M_alloc):
        Xs, Ps = region_measure(region)
        print(f"[B block]    {region['name']:15s} N={region['n']:6d} M={M_r}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r)
        Xb_list.append(Xh.detach())
        Pb_list.append(Ph.detach())
    Xb = torch.cat(Xb_list, dim=0)
    Pb = torch.cat(Pb_list, dim=0)

    # 2) classify optimized block points (on GPU): core (fixed) vs vstrip vs hstrip
    #    (vstrip owns the crossings, so hstrip excludes the vertical-seam neighbourhood).
    dv, iv = nearest_line(Xb[:, 0], vlines_g)
    dh, ih = nearest_line(Xb[:, 1], hlines_g)
    near_v, near_h = dv <= h, dh <= h
    core = ~near_v & ~near_h

    # original data, classified the same way but ON CPU (targets for the refinement).
    dvo, ivo = nearest_line(X_orig[:, 0], vlines_c)
    dho, iho = nearest_line(X_orig[:, 1], hlines_c)
    near_vo, near_ho = dvo <= h, dho <= h

    def orig_band(mask):
        # pull one seam band out of the CPU-resident original cloud onto the GPU.
        return X_orig[mask].to(device), P_orig[mask].to(device)

    next_id = 0
    def emit(Xh, Ph):
        global next_id
        if Xh.shape[0] == 0:
            return
        X_hat_list.append(Xh.detach())
        P_hat_list.append(Ph.detach())
        region_ids.append(torch.full((Xh.shape[0],), next_id, dtype=torch.int32))
        next_id += 1

    # 3a) core: block optimum kept verbatim.
    print(f"[B core]     N={int(core.sum())}")
    emit(Xb[core], Pb[core])

    # 3b) vertical seams (own the crossings).
    for i in range(vlines_g.shape[0]):
        mov = near_v & (iv == i)
        tgt = near_vo & (ivo == i)
        if mov.sum() == 0:
            continue
        Xm, Pm = Xb[mov], Pb[mov]
        alpha = (1.0 - (Xm[:, 0] - vlines_g[i]).abs() / h).clamp(0.0, 1.0)
        print(f"[B vstrip{i:2d}] mov={int(mov.sum()):5d} tgt={int(tgt.sum()):6d}")
        emit(*refine_strip(orig_band(tgt), Xm, Pm, alpha))

    # 3c) horizontal seams (clipped away from the vertical seams).
    for j in range(hlines_g.shape[0]):
        mov = near_h & ~near_v & (ih == j)
        tgt = near_ho & ~near_vo & (iho == j)
        if mov.sum() == 0:
            continue
        Xm, Pm = Xb[mov], Pb[mov]
        alpha = (1.0 - (Xm[:, 1] - hlines_g[j]).abs() / h).clamp(0.0, 1.0)
        print(f"[B hstrip{j:2d}] mov={int(mov.sum()):5d} tgt={int(tgt.sum()):6d}")
        emit(*refine_strip(orig_band(tgt), Xm, Pm, alpha))
else:
    counts = [r["X"].shape[0] for r in regions]
    total_n = sum(counts)
    M_alloc = [max(1, min(round(M_TOTAL * c / total_n), c)) for c in counts]

    for rid, (region, M_r) in enumerate(zip(regions, M_alloc)):
        Xs, Ps = region["X"].to(device), region["P"].to(device)
        print(f"[{rid:2d}] {region['name']:13s} kind={region['kind']:6s} "
              f"N={Xs.shape[0]:6d}  M={M_r}")

        Xh, Ph = optimize_measure((Xs, Ps), M_r)
        X_hat_list.append(Xh)
        P_hat_list.append(Ph)
        region_ids.append(torch.full((Xh.shape[0],), rid, dtype=torch.int32))

total_time = time.time() - t_start
print(f"Total reduction time      : {total_time:.2f} s")
# ── Merge ──────────────────────────────────────────────
# back to CPU to match the CPU-resident X_min/X_max used by the exporters.
X_hat_all = torch.cat(X_hat_list, dim=0).cpu()
P_hat_all = torch.cat(P_hat_list, dim=0).cpu()
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
