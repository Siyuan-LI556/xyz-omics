# scripts/barseq_varifold_tiled.py
import os
import time
import torch
from src.config import (
    SUFFIX, N_GRID, strip_width, TILE_MODE, EXPAND_HALO, Input,
)
from src.io.loader import load_barseq
from src.io.vtk_export import export_orig_vtp, export_hat_vtp
from src.preprocessing.tiling import (
    split_slice_grid, reduce_blocks_only, reduce_blocks_and_strips,
)
from src.preprocessing.tiling_expanded import (
    split_slice_grid_expanded, reduce_blocks_and_overlaps, reduce_blocks_expanded,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Tiling: {N_GRID}x{N_GRID}, strip_width={strip_width}, mode={TILE_MODE}")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTK_FILE   = os.path.join(BASE_DIR, "data", "BARSeq", Input)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load + tile ────────────────────────────────────────
# Keep the (large) original cloud resident on CPU — only per-region / per-strip subsets
# are moved to `device` inside the reducers.
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(VTK_FILE, torch.device("cpu"))
del W_orig, P_norm_orig
if device.type == "cuda":
    torch.cuda.empty_cache()

REDUCERS = {
    "blocks_only":         reduce_blocks_only,
    "blocks_and_strips":   reduce_blocks_and_strips,
    "blocks_and_overlaps": reduce_blocks_and_overlaps,
    "blocks_expanded":     reduce_blocks_expanded,
}
if TILE_MODE not in REDUCERS:
    raise ValueError(f"Unknown TILE_MODE: '{TILE_MODE}'.")

# Plans A/B share the plain block partition; plans B'/C need the halo partition.
if TILE_MODE in ("blocks_only", "blocks_and_strips"):
    regions = split_slice_grid(X_orig, P_orig, n=N_GRID)
else:
    regions = split_slice_grid_expanded(X_orig, P_orig, n=N_GRID, mode=TILE_MODE,
                                        strip_width=strip_width, halo=EXPAND_HALO)

# ── Per-region reduction ───────────────────────────────
t_start = time.time()
X_hat_list, P_hat_list, region_ids = REDUCERS[TILE_MODE](regions, X_orig, P_orig, device)
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
