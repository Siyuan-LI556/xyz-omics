# src/preprocessing/tiling.py
import torch

from src import config
from src.losses.varifold import make_varifold_fn
from src.subsampling import subsample
from src.optim import run_optimizer


# ═══════════════════════════════════════════════════════════════════════════
# Grid primitives (shared with tiling_expanded.py)
# ═══════════════════════════════════════════════════════════════════════════


def grid_edges(n, bounds=None):
    """Uniform n x n cell edges over the domain. Returns (xb, yb, ex, ey)."""
    xb, yb = ((0.0, 1.0), (0.0, 1.0)) if bounds is None else bounds
    ex = [xb[0] + (xb[1] - xb[0]) * k / n for k in range(n + 1)]
    ey = [yb[0] + (yb[1] - yb[0]) * k / n for k in range(n + 1)]
    return xb, yb, ex, ey


def block_masks(x, y, n, ex, ey):
    """
    Column/row masks of the n x n block partition. Cells are half-open (the last one
    closed), so the blocks tile the domain with no overlap and no gap.
    """
    col_masks = [
        (x >= ex[c]) & ((x <= ex[c + 1]) if c == n - 1 else (x < ex[c + 1]))
        for c in range(n)
    ]
    row_masks = [
        (y >= ey[r]) & ((y <= ey[r + 1]) if r == n - 1 else (y < ey[r + 1]))
        for r in range(n)
    ]
    return col_masks, row_masks


def add_region(regions, name, kind, mask, **meta):
    """
    Append one region to `regions`, skipping it when it holds no point. A region only
    stores indices into the original cloud (never a copy) — region_measure pulls the
    points onto the compute device when they are actually needed.
    """
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    regions.append({"name": name, "kind": kind, "idx": idx, "n": idx.numel(), **meta})


def strip_geometry(n, strip_width=0.04, bounds=None):
    """
    Grid geometry shared by the post-optimization strip-refinement pipeline.

    Returns the interior grid lines (the block seams) and the strip half-width, so
    the caller can carve thin seam bands out of an *already optimized* point cloud.

    Returns a dict with:
        vlines : (n-1,) interior vertical seam positions (x = ex[1..n-1])
        hlines : (n-1,) interior horizontal seam positions (y = ey[1..n-1])
        h      : strip half-width (strip_width / 2)
        bounds : ((xb0, xb1), (yb0, yb1))
    """
    xb, yb, ex, ey = grid_edges(n, bounds)
    return {
        "vlines": ex[1:n],
        "hlines": ey[1:n],
        "h": strip_width / 2.0,
        "bounds": (xb, yb),
    }


def split_slice_grid(X, P, n=4, bounds=None):
    """
    Plain n x n block partition — used by BOTH plan A (blocks_only) and plan B
    (blocks_and_strips): the two plans split the data the same way and differ only in
    how the optimized cloud is post-processed.

    For plan B the seam strips are carved out of the *optimized* block cloud afterwards
    (see strip_geometry + reduce_blocks_and_strips), so blocks and strips end up
    complementary and non-overlapping instead of the strips being overlaid on raw data.
    """
    x = X[:, 0].detach().cpu().contiguous()
    y = X[:, 1].detach().cpu().contiguous()
    _, _, ex, ey = grid_edges(n, bounds)
    col_masks, row_masks = block_masks(x, y, n, ex, ey)

    regions = []
    for r in range(n):
        for c in range(n):
            add_region(regions, f"block_r{r}_c{c}", "block", row_masks[r] & col_masks[c])
    return regions


# ═══════════════════════════════════════════════════════════════════════════
# Per-region reduction pipeline (helpers shared with tiling_expanded.py)
#
# Config, subsampling, the varifold factory and the optimizer dispatch are all
# reused from their own feature modules.
# ═══════════════════════════════════════════════════════════════════════════


def tiled_varifold_fn(dim=3):
    """Varifold callable for the tiled pipeline (isotropic bandwidth = BASE_SIGMA).
    `dim` is the ambient dimension of the positions (X.shape[1])."""
    return make_varifold_fn(config.KERNEL_TYPE, config.M, config.BASE_SIGMA,
                            config.SIGMA_XY, config.SIGMA_Z, dim=dim)


def alloc_budget(counts, M_total):
    """Split the global budget M_total across regions, proportional to point count."""
    total = sum(counts)
    return [max(1, min(round(M_total * c / total), c)) for c in counts]


def region_measure(X_orig, P_orig, region, device):
    """Pull one region's points from the CPU-resident cloud onto `device`."""
    idx = region["idx"]
    return X_orig[idx].to(device), P_orig[idx].to(device)


def emit_region(X_list, P_list, id_list, Xh, Ph, rid):
    """
    Append one optimized region to the output lists, tagged with `rid`. Empty regions
    are skipped. Returns True when something was appended.
    """
    if Xh.shape[0] == 0:
        return False
    X_list.append(Xh.detach())
    P_list.append(Ph.detach())
    id_list.append(torch.full((Xh.shape[0],), rid, dtype=torch.int32))
    return True


def nearest_line(coord, lines):
    """Nearest interior grid line to each coord (uniform grid → round, no (N,L) temp)."""
    step = (lines[-1] - lines[0]) / (lines.shape[0] - 1) if lines.shape[0] > 1 else 1.0
    k = torch.round((coord - lines[0]) / step).clamp_(0, lines.shape[0] - 1).long()
    return (coord - lines[k]).abs(), k


def initial_measure(Xs, Ps, M_r, device, init=None):
    """Initial representative set: reuse `init` if big enough, else subsample."""
    if init is not None and init[0].shape[0] >= M_r:
        X_src, P_src = init
    else:
        X_src, P_src = Xs, Ps

    if M_r >= X_src.shape[0]:
        # Bypasses subsample(), so FREEZE_P has to be applied here too.
        return (X_src.detach().clone().requires_grad_(True),
                P_src.detach().clone().requires_grad_(not config.FREEZE_P))
    return subsample(config.SUBSAMPLE_METHOD, X_src.detach(), P_src.detach(), M_r, device)


def optimize_measure(S_target, M_r, device, init=None):
    """Reduce one measure S_target to ~M_r representative points and optimize it."""
    Xs, Ps = S_target
    Ns = Xs.shape[0]
    if Ns == 0:
        return Xs[:0].detach(), Ps[:0].detach()
    M_r = max(1, min(int(M_r), Ns))

    if M_r >= Ns and init is None:
        return Xs.detach().clone(), Ps.detach().clone()

    Xh, Ph = initial_measure(Xs, Ps, M_r, device, init=init)
    vf = tiled_varifold_fn(dim=Xs.shape[1])
    Xh, Ph, _, _ = run_optimizer(S_target, Xh, Ph, vf)
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
    P_t = P_hat.clone().requires_grad_(config.STRIP_MOVE_P)
    vf = tiled_varifold_fn(dim=X_hat.shape[1])
    X_t, P_t, _, _ = run_optimizer(S_target, X_t, P_t, vf)

    rho = alpha.detach().view(-1, 1)                       # (M,1), in [0,1]
    X_sharp = X_hat + rho * (X_t.detach() - X_hat)         # X# = X_hat + rho*(X_tilde-X_hat)
    if config.STRIP_MOVE_P:
        P_sharp = (P_hat + rho * (P_t.detach() - P_hat)).clamp(min=0.0)
    else:
        P_sharp = P_hat
    return X_sharp.detach(), P_sharp.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Plan A / Plan B reducers
# ═══════════════════════════════════════════════════════════════════════════


def reduce_blocks_only(regions, X_orig, P_orig, device):
    """Plan A: reduce each block independently, no seam handling."""
    X_hat_list, P_hat_list, region_ids = [], [], []
    M_alloc = alloc_budget([r["n"] for r in regions], config.M_TOTAL)

    for rid, (region, M_r) in enumerate(zip(regions, M_alloc)):
        Xs, Ps = region_measure(X_orig, P_orig, region, device)
        print(f"[{rid:2d}] {region['name']:13s} kind={region['kind']:6s} "
              f"N={region['n']:6d}  M={M_r}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r, device)
        emit_region(X_hat_list, P_hat_list, region_ids, Xh, Ph, rid)
    return X_hat_list, P_hat_list, region_ids


def reduce_blocks_and_strips(regions, X_orig, P_orig, device):
    """
    Plan B (revised): optimize the block partition first, then carve thin seam bands
    out of that optimized cloud and re-optimize them against the ORIGINAL data with the
    block junctions blended in (hat weight). Blocks-minus-seams and refined seams are
    complementary, so the merge is non-overlapping and point-count-preserving.
    """
    X_hat_list, P_hat_list, region_ids = [], [], []
    geom = strip_geometry(config.N_GRID, strip_width=config.strip_width)
    h = geom["h"]
    # Lines live on two devices: GPU copy classifies the (small) optimized block cloud;
    # CPU copy classifies the (large) original cloud without moving it to the GPU.
    vlines_g = torch.tensor(geom["vlines"], device=device)
    hlines_g = torch.tensor(geom["hlines"], device=device)
    vlines_c = vlines_g.cpu()
    hlines_c = hlines_g.cpu()

    # 1) optimize each block (full partition), same budget split as blocks_only.
    M_alloc = alloc_budget([r["n"] for r in regions], config.M_TOTAL)
    Xb_list, Pb_list = [], []
    for region, M_r in zip(regions, M_alloc):
        Xs, Ps = region_measure(X_orig, P_orig, region, device)
        print(f"[B block]    {region['name']:15s} N={region['n']:6d} M={M_r}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r, device)
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
        nonlocal next_id
        if emit_region(X_hat_list, P_hat_list, region_ids, Xh, Ph, next_id):
            next_id += 1

    # 3a) core: block optimum kept verbatim.
    print(f"[B core]     N={int(core.sum())}")
    emit(Xb[core], Pb[core])

    # 3b) vertical seams (own the crossings), then 3c) horizontal seams (clipped away
    #     from the vertical ones). Same refinement, only the axis and the masks differ.
    seams = [
        ("vstrip", 0, vlines_g, near_v, iv, near_vo, ivo),
        ("hstrip", 1, hlines_g, near_h & ~near_v, ih, near_ho & ~near_vo, iho),
    ]
    for tag, axis, lines, mov_near, mov_idx, tgt_near, tgt_idx in seams:
        for i in range(lines.shape[0]):
            mov = mov_near & (mov_idx == i)
            tgt = tgt_near & (tgt_idx == i)
            if mov.sum() == 0:
                continue
            Xm, Pm = Xb[mov], Pb[mov]
            alpha = (1.0 - (Xm[:, axis] - lines[i]).abs() / h).clamp(0.0, 1.0)
            print(f"[B {tag}{i:2d}] mov={int(mov.sum()):5d} tgt={int(tgt.sum()):6d}")
            emit(*refine_strip(orig_band(tgt), Xm, Pm, alpha))
    return X_hat_list, P_hat_list, region_ids
