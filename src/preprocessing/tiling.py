# src/preprocessing/tiling.py
import torch


def split_slice_grid(X, P, n=4, strip_width=0.04, bounds=None, mode="blocks_only"):

    device = X.device
    x, y = X[:, 0].contiguous(), X[:, 1].contiguous()
    xb, yb = ((0.0, 1.0), (0.0, 1.0)) if bounds is None else bounds

    regions = []

    def add(name, kind, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        regions.append({"name": name, "kind": kind,
                        "idx": idx, "X": X[idx], "P": P[idx]})


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

    if mode == "blocks_and_strips":
        for r in range(n):
            for c in range(n):
                add(f"block_r{r}_c{c}", "block", row_masks[r] & col_masks[c])

        for i, line in enumerate(gx):
            add(f"vstrip_{i}", "vstrip", (x >= line - h) & (x <= line + h))
        for j, line in enumerate(gy):
            add(f"hstrip_{j}", "hstrip", (y >= line - h) & (y <= line + h))
        return regions