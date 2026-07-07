# src/preprocessing/tiling.py
import torch


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
    xb, yb = ((0.0, 1.0), (0.0, 1.0)) if bounds is None else bounds
    ex = [xb[0] + (xb[1] - xb[0]) * k / n for k in range(n + 1)]
    ey = [yb[0] + (yb[1] - yb[0]) * k / n for k in range(n + 1)]
    return {
        "vlines": ex[1:n],
        "hlines": ey[1:n],
        "h": strip_width / 2.0,
        "bounds": (xb, yb),
    }


def split_slice_grid(X, P, n=4, strip_width=0.04, bounds=None, mode="blocks_only"):

    x = X[:, 0].detach().cpu().contiguous()
    y = X[:, 1].detach().cpu().contiguous()
    xb, yb = ((0.0, 1.0), (0.0, 1.0)) if bounds is None else bounds
    keep_data = mode != "blocks_and_overlaps"

    regions = []

    def add(name, kind, mask, **meta):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        region = {"name": name, "kind": kind, "idx": idx, "n": idx.numel(), **meta}
        if keep_data:
            idx_dev = idx.to(X.device)
            region["X"] = X[idx_dev]
            region["P"] = P[idx_dev]
        regions.append(region)


    ex = [xb[0] + (xb[1] - xb[0]) * k / n for k in range(n + 1)]
    ey = [yb[0] + (yb[1] - yb[0]) * k / n for k in range(n + 1)]
    gx = ex[1:n]
    gy = ey[1:n]
    h  = strip_width / 2.0

    col_masks = [
        (x >= ex[c]) & ((x <= ex[c + 1]) if c == n - 1 else (x < ex[c + 1]))
        for c in range(n)
    ]
    row_masks = [
        (y >= ey[r]) & ((y <= ey[r + 1]) if r == n - 1 else (y < ey[r + 1]))
        for r in range(n)
    ]

    #plan A, blocks only
    if mode == "blocks_only":
        for r in range(n):
            for c in range(n):
                add(f"block_r{r}_c{c}", "block", row_masks[r] & col_masks[c])
        return regions

    # plan B, blocks and strips.
    # Only the block partition is produced here. The seam strips are carved out of the
    # *optimized* block cloud afterwards (see strip_geometry + the refinement pipeline in
    # scripts/barseq_varifold_tiled.py), so blocks and strips end up complementary and
    # non-overlapping instead of the strips being overlaid on the raw data.
    if mode == "blocks_and_strips":
        for r in range(n):
            for c in range(n):
                add(f"block_r{r}_c{c}", "block", row_masks[r] & col_masks[c])
        return regions

    # plan B', expanded blocks for boundary context + shared pairwise overlaps.
    if mode == "blocks_and_overlaps":
        rect = lambda x0, x1, y0, y1: (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        overlap_half = h / 2.0

        for r in range(n):
            for c in range(n):
                bid = r * n + c
                add(f"block_r{r}_c{c}", "block", row_masks[r] & col_masks[c],
                    block_id=bid, r=r, c=c,
                    bounds=(ex[c], ex[c + 1], ey[r], ey[r + 1]))

                x0, x1 = max(xb[0], ex[c] - h), min(xb[1], ex[c + 1] + h)
                y0, y1 = max(yb[0], ey[r] - h), min(yb[1], ey[r + 1] + h)
                add(f"expanded_r{r}_c{c}", "expanded", rect(x0, x1, y0, y1),
                    block_id=bid, r=r, c=c, bounds=(x0, x1, y0, y1))

        # vertical seam ex[c]: overlap of horizontally-adjacent blocks (r,c-1)-(r,c)
        for c in range(1, n):
            for r in range(n):
                left_id = r * n + (c - 1)
                right_id = r * n + c
                x0, x1 = ex[c] - overlap_half, ex[c] + overlap_half
                y0, y1 = ey[r], ey[r + 1]
                add(f"voverlap_r{r}_c{c}", "overlap",
                    rect(x0, x1, y0, y1),
                    u=left_id, v=right_id, orientation="vertical",
                    seam=ex[c], bounds=(x0, x1, y0, y1))

        # horizontal seam ey[r]: overlap of vertically-adjacent blocks (r-1,c)-(r,c)
        for r in range(1, n):
            for c in range(n):
                lower_id = (r - 1) * n + c
                upper_id = r * n + c
                x0, x1 = ex[c], ex[c + 1]
                y0, y1 = ey[r] - overlap_half, ey[r] + overlap_half
                add(f"hoverlap_r{r}_c{c}", "overlap",
                    rect(x0, x1, y0, y1),
                    u=lower_id, v=upper_id, orientation="horizontal",
                    seam=ey[r], bounds=(x0, x1, y0, y1))
        return regions
