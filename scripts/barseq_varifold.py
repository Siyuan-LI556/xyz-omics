# scripts/barseq_varifold.py
import os
import torch
from src.config import (
    SUBSAMPLE_METHOD, KERNEL_TYPE, OPTIMIZER, SUFFIX, Input,
    M, BASE_SIGMA, SIGMA_XY, SIGMA_Z,
)
from src.io.loader import load_barseq, load_middle_slices
from src.subsampling import subsample
from src.optim import run_optimizer
from src.losses.varifold import make_varifold_fn
from src.io.vtk_export import export_orig_vtp, export_hat_vtp, export_middle_slices_vtp, export_hat_middle_slices_vtp
from src.io.plot import plot_loss_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run config  : {SUFFIX}")

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", Input)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(DATA_FILE, device)
S = (X_orig, P_orig)

# ── Subsample ──────────────────────────────────────────
print(f"Subsampling method: {SUBSAMPLE_METHOD}  →  {M} representative points")
X_hat, P_hat = subsample(SUBSAMPLE_METHOD, X_orig, P_orig, M, device)

# ── Build varifold function ────────────────────────────
print(f"Kernel type: {KERNEL_TYPE}")
if KERNEL_TYPE == "isotropic":
    print(f"Isotropic bandwidth: {BASE_SIGMA * (1 / M) ** (1 / 3):.4f}")
elif KERNEL_TYPE == "anisotropic":
    print(f"Anisotropic bandwidth: sigma_xy={SIGMA_XY}, sigma_z={SIGMA_Z}")
varifold_fn = make_varifold_fn(KERNEL_TYPE, M, BASE_SIGMA, SIGMA_XY, SIGMA_Z)

# ── Optimize ───────────────────────────────────────────
print(f"Optimizer: {OPTIMIZER}")
X_hat, P_hat, loss_history, time_history = run_optimizer(S, X_hat, P_hat, varifold_fn)

# ── Export ─────────────────────────────────────────────
export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
export_hat_vtp(X_hat, P_hat, X_min, X_max, OUTPUT_DIR, suffix=SUFFIX)
'''
torch.save(
    (X_orig, X_hat, P_hat, P_orig, loss_history),
    os.path.join(OUTPUT_DIR, f"results_{SUFFIX}.pt"),
)
#plot_loss_curve(loss_history, time_history, output_dir=OUTPUT_DIR, suffix=SUFFIX)

# ── Middle 3 slices comparison ─────────────────────────
X_mid, P_mid, selected_z, slice_id = load_middle_slices(DATA_FILE, n=3)
#export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, OUTPUT_DIR)
#export_hat_middle_slices_vtp(X_hat, P_hat, X_min, X_max, selected_z, OUTPUT_DIR, suffix=SUFFIX)

S_full    = (X_orig, P_orig)
S_hat_all = (X_hat, P_hat)
with torch.no_grad():
    term0   = varifold_fn(S_full, S_full)          # <S, S>
    term1   = varifold_fn(S_hat_all, S_hat_all)    # <Ŝ, Ŝ>
    term2   = varifold_fn(S_full, S_hat_all)       # <S, Ŝ>
    dist_sq = (term0 + term1 - 2 * term2).item()

dist = dist_sq ** 0.5 if dist_sq > 0 else float("nan")

print("──────────────────────────────────────────")
print(f"varifold distance² : {dist_sq:.6e}")
print(f"varifold distance  : {dist:.6e}")
print("──────────────────────────────────────────")
'''