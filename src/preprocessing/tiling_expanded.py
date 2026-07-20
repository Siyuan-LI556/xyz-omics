# src/preprocessing/tiling_expanded.py
"""
Halo-based tiling plans: B' (blocks_and_overlaps) and C (blocks_expanded).

Both optimize each block together with a surrounding halo, so the varifold kernel does
not see an artificial block edge. They then recover a non-overlapping result in
different ways: C slices the optimized cloud back to the true block bounds, while B'
keeps the sliced cores and re-optimizes the shared seam overlaps against BOTH
neighbouring blocks at once.

Grid primitives and reduction helpers are reused from tiling.py.
"""
import torch

from src import config
from src.optim import run_optimizer_joint
from src.preprocessing.tiling import (
    grid_edges, block_masks, add_region,
    alloc_budget, region_measure, initial_measure, optimize_measure,
    tiled_varifold_fn, emit_region,
)


def rect_mask(x, y, x0, x1, y0, y1):
    """Closed rectangle mask — halo and overlap regions are allowed to share borders."""
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def slice_measure(S, bounds, orientation=None, seam=None, k=1):
    """
    Mask a measure S=(X, P) by a bounding box. If nothing falls inside and an
    orientation/seam is given, fall back to the k points nearest the seam.
    """
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


def slice_to_core(X, P, bounds, r, c, n):
    """Half-open slice of a cloud back to the true block cell (same rule as block_masks)."""
    x0, x1, y0, y1 = bounds
    mx = (X[:, 0] >= x0) & ((X[:, 0] <= x1) if c == n - 1 else (X[:, 0] < x1))
    my = (X[:, 1] >= y0) & ((X[:, 1] <= y1) if r == n - 1 else (X[:, 1] < y1))
    m = mx & my
    return X[m], P[m]


def split_slice_grid_expanded(X, P, n=4, mode="blocks_expanded", strip_width=0.04,
                              halo=0.02, bounds=None):
    """
    Partition for the halo-based plans.

    mode="blocks_expanded" (plan C):
        true block cores + block+halo regions (bounds = the TRUE block, for slicing back).
    mode="blocks_and_overlaps" (plan B'):
        true block cores + block+h regions + pairwise seam overlaps, with h = strip_width/2.
    """
    x = X[:, 0].detach().cpu().contiguous()
    y = X[:, 1].detach().cpu().contiguous()
    xb, yb, ex, ey = grid_edges(n, bounds)
    col_masks, row_masks = block_masks(x, y, n, ex, ey)
    h = strip_width / 2.0

    regions = []

    # plan C, expanded blocks only (extends B' without the overlap seam step):
    # optimize each block together with a halo of surrounding context, then slice the
    # optimized cloud back to the TRUE block bounds and merge. The halo cancels the kernel
    # edge-deficit, so adjacent cores meet at the seam without holes.
    if mode == "blocks_expanded":
        for r in range(n):
            for c in range(n):
                bid = r * n + c
                core_bounds = (ex[c], ex[c + 1], ey[r], ey[r + 1])
                add_region(regions, f"block_r{r}_c{c}", "block",
                           row_masks[r] & col_masks[c],
                           block_id=bid, r=r, c=c, bounds=core_bounds)

                x0, x1 = max(xb[0], ex[c] - halo), min(xb[1], ex[c + 1] + halo)
                y0, y1 = max(yb[0], ey[r] - halo), min(yb[1], ey[r + 1] + halo)
                add_region(regions, f"expanded_r{r}_c{c}", "expanded",
                           rect_mask(x, y, x0, x1, y0, y1),
                           block_id=bid, r=r, c=c, bounds=core_bounds)  # TRUE block, for slicing
        return regions

    # plan B', expanded blocks for boundary context + shared pairwise overlaps.
    if mode == "blocks_and_overlaps":
        overlap_half = h / 2.0

        for r in range(n):
            for c in range(n):
                bid = r * n + c
                add_region(regions, f"block_r{r}_c{c}", "block",
                           row_masks[r] & col_masks[c],
                           block_id=bid, r=r, c=c,
                           bounds=(ex[c], ex[c + 1], ey[r], ey[r + 1]))

                x0, x1 = max(xb[0], ex[c] - h), min(xb[1], ex[c + 1] + h)
                y0, y1 = max(yb[0], ey[r] - h), min(yb[1], ey[r + 1] + h)
                add_region(regions, f"expanded_r{r}_c{c}", "expanded",
                           rect_mask(x, y, x0, x1, y0, y1),
                           block_id=bid, r=r, c=c, bounds=(x0, x1, y0, y1))

        # vertical seam ex[c]: overlap of horizontally-adjacent blocks (r,c-1)-(r,c)
        for c in range(1, n):
            for r in range(n):
                left_id = r * n + (c - 1)
                right_id = r * n + c
                x0, x1 = ex[c] - overlap_half, ex[c] + overlap_half
                y0, y1 = ey[r], ey[r + 1]
                add_region(regions, f"voverlap_r{r}_c{c}", "overlap",
                           rect_mask(x, y, x0, x1, y0, y1),
                           u=left_id, v=right_id, orientation="vertical",
                           seam=ex[c], bounds=(x0, x1, y0, y1))

        # horizontal seam ey[r]: overlap of vertically-adjacent blocks (r-1,c)-(r,c)
        for r in range(1, n):
            for c in range(n):
                lower_id = (r - 1) * n + c
                upper_id = r * n + c
                x0, x1 = ex[c], ex[c + 1]
                y0, y1 = ey[r] - overlap_half, ey[r] + overlap_half
                add_region(regions, f"hoverlap_r{r}_c{c}", "overlap",
                           rect_mask(x, y, x0, x1, y0, y1),
                           u=lower_id, v=upper_id, orientation="horizontal",
                           seam=ey[r], bounds=(x0, x1, y0, y1))
        return regions

    raise ValueError(f"Unknown expanded TILE_MODE: '{mode}'.")


def reduce_blocks_and_overlaps(regions, X_orig, P_orig, device):
    """
    Plan B': expanded blocks for boundary context + shared pairwise overlaps.
    Optimize each expanded block, then either emit the expanded cloud directly, or
    (BPRIME_OPTIMIZE_OVERLAP) slice the true cores out of it and jointly optimize the
    seam overlaps against both neighbouring blocks.
    """
    X_hat_list, P_hat_list, region_ids = [], [], []
    blocks = [r for r in regions if r["kind"] == "block"]
    expanded = [r for r in regions if r["kind"] == "expanded"]
    overlaps = [r for r in regions if r["kind"] == "overlap"]

    M_alloc = alloc_budget([r["n"] for r in expanded], config.M_TOTAL)

    expanded_hat = {}
    for region, M_r in zip(expanded, M_alloc):
        Xs, Ps = region_measure(X_orig, P_orig, region, device)
        print(f"[B' expanded] {region['name']:15s} N={region['n']:6d} M={M_r}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r, device)
        del Xs, Ps
        expanded_hat[region["block_id"]] = (Xh.detach(), Ph.detach())

        if not config.BPRIME_OPTIMIZE_OVERLAP:
            emit_region(X_hat_list, P_hat_list, region_ids, Xh, Ph, region["block_id"])

    if config.BPRIME_OPTIMIZE_OVERLAP:
        for region in blocks:
            Xh, Ph = slice_measure(expanded_hat[region["block_id"]], region["bounds"])
            print(f"[B' core]     {region['name']:15s} from expanded M={Xh.shape[0]}")
            emit_region(X_hat_list, P_hat_list, region_ids, Xh, Ph, region["block_id"])

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
            M_r = int(n_src * config.BPRIME_OVERLAP_KEEP_RATIO)
            if M_r == 0:
                continue
            M_r = max(1, min(M_r, n_src))
            print(f"[B' overlap]  {region['name']:15s} M={M_r} "
                  f"left={left[0].shape[0]} right={right[0].shape[0]}")

            X_src = torch.cat([S[0].detach() for S in (left, right) if S[0].shape[0] > 0], dim=0)
            P_src = torch.cat([S[1].detach() for S in (left, right) if S[0].shape[0] > 0], dim=0)
            Xh, Ph = initial_measure(X_src, P_src, M_r, device)
            vf = tiled_varifold_fn()
            Xh, Ph, _, _ = run_optimizer_joint([left, right], Xh, Ph, vf)

            emit_region(X_hat_list, P_hat_list, region_ids, Xh, Ph, len(blocks) + rid)
    return X_hat_list, P_hat_list, region_ids


def reduce_blocks_expanded(regions, X_orig, P_orig, device):
    """
    Plan C: optimize each block WITH a halo of surrounding context, then slice the
    optimized cloud back to the true (original) block bounds and merge. Non-overlapping
    cores; the halo removes the kernel edge-deficit so adjacent cores meet without holes.
    """
    X_hat_list, P_hat_list, region_ids = [], [], []
    n = config.N_GRID
    blocks = {r["block_id"]: r for r in regions if r["kind"] == "block"}
    expanded = [r for r in regions if r["kind"] == "expanded"]
    total_core = sum(r["n"] for r in blocks.values())

    for region in expanded:
        bid = region["block_id"]
        core = blocks.get(bid)          # missing → true block cell has no original points
        exp_n = region["n"]
        if core is None or exp_n == 0:
            continue                     # its halo mass is represented by neighbouring cores
        core_n = core["n"]
        # Budget the expanded optimization so the sliced-back core ≈ its fair share of M_TOTAL:
        # optimize M_exp points over the expanded region, keep ~core/exp of them after slicing.
        M_core = max(1, round(config.M_TOTAL * core_n / max(total_core, 1)))
        M_r = max(1, min(round(M_core * exp_n / max(core_n, 1)), exp_n))
        Xs, Ps = region_measure(X_orig, P_orig, region, device)
        print(f"[C expand]  {region['name']:15s} exp_N={exp_n:6d} M={M_r} core_N={core_n}")
        Xh, Ph = optimize_measure((Xs, Ps), M_r, device)
        Xc, Pc = slice_to_core(Xh.detach(), Ph.detach(),
                               region["bounds"], region["r"], region["c"], n)
        emit_region(X_hat_list, P_hat_list, region_ids, Xc, Pc, bid)
    return X_hat_list, P_hat_list, region_ids
