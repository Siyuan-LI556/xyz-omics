# scripts/barseq_varifold.py
import os
import torch
from src.config import (
    RUN_ID, SUBSAMPLE_METHOD, KERNEL_TYPE, OPTIMIZER, SUFFIX,
    M, BASE_SIGMA, SIGMA_XY, SIGMA_Z,
    LR, MAX_ITER, HISTORY_SIZE, EPOCHS, TOL, PATIENCE,
    ADAM_LR_X, ADAM_LR_P, ADAM_EPOCHS, ADAM_TOL, ADAM_PATIENCE,
)
from src.io.loader import load_barseq, load_middle_slices
from src.subsampling.kmeans import kmeans_subsample
from src.subsampling.random import random_subsample
from src.optim.LBFGS import optimize_lbfgs
from src.optim.Adam import optimize_adam
from src.losses.varifold import varifold_sp, varifold_sp_anisotropic
from src.io.vtk_export import export_orig_vtp, export_hat_vtp, export_middle_slices_vtp, export_hat_middle_slices_vtp
from src.io.plot import plot_loss_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run config  : {SUFFIX}")

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "npz_slices","slice_10.npz")
#DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq" ,"all_slices_C57BL6J.npz")
#DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "D076_1L_approx200um.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(DATA_FILE, device)
S = (X_orig, P_orig)

# ── Subsample ──────────────────────────────────────────
print(f"Subsampling method: {SUBSAMPLE_METHOD}  →  {M} representative points")
if SUBSAMPLE_METHOD == "kmeans":
    X_hat, P_hat = kmeans_subsample(X_orig, P_orig, M, device)
elif SUBSAMPLE_METHOD == "random":
    X_hat, P_hat = random_subsample(X_orig, P_orig, M)
else:
    raise ValueError(f"Unknown SUBSAMPLE_METHOD: '{SUBSAMPLE_METHOD}'. Use 'kmeans' or 'random'.")

# ── Build varifold function ────────────────────────────
print(f"Kernel type: {KERNEL_TYPE}")
if KERNEL_TYPE == "isotropic":
    bandwidth = BASE_SIGMA * (1 / M) ** (1 / 3)
    print(f"Isotropic bandwidth: {bandwidth:.4f}")
    varifold_fn = lambda S1, S2: varifold_sp(S1, S2, bandwidth)
elif KERNEL_TYPE == "anisotropic":
    print(f"Anisotropic bandwidth: sigma_xy={SIGMA_XY}, sigma_z={SIGMA_Z}")
    varifold_fn = lambda S1, S2: varifold_sp_anisotropic(S1, S2, SIGMA_XY, SIGMA_Z)
else:
    raise ValueError(f"Unknown KERNEL_TYPE: '{KERNEL_TYPE}'. Use 'isotropic' or 'anisotropic'.")

# ── Optimize ───────────────────────────────────────────
print(f"Optimizer: {OPTIMIZER}")
if OPTIMIZER == "lbfgs":
    X_hat, P_hat, loss_history, time_history = optimize_lbfgs(
        S, X_hat, P_hat, varifold_fn,
        lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE,
        epochs=EPOCHS, tol=TOL, patience=PATIENCE,
    )
elif OPTIMIZER == "adam":
    X_hat, P_hat, loss_history, time_history = optimize_adam(
        S, X_hat, P_hat, varifold_fn,
        lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
        epochs=ADAM_EPOCHS, tol=ADAM_TOL, patience=ADAM_PATIENCE,
    )
else:
    raise ValueError(f"Unknown OPTIMIZER: '{OPTIMIZER}'. Use 'lbfgs' or 'adam'.")

# ── Export ─────────────────────────────────────────────
export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
export_hat_vtp(X_hat, P_hat, X_min, X_max, OUTPUT_DIR, suffix=SUFFIX)
torch.save(
    (X_orig, X_hat, P_hat, P_orig, loss_history),
    os.path.join(OUTPUT_DIR, f"results_{SUFFIX}.pt"),
)
#plot_loss_curve(loss_history, time_history, output_dir=OUTPUT_DIR, suffix=SUFFIX)
'''
# ── Middle 3 slices comparison ─────────────────────────
X_mid, P_mid, selected_z, slice_id = load_middle_slices(DATA_FILE, n=3)
export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, OUTPUT_DIR)
export_hat_middle_slices_vtp(X_hat, P_hat, X_min, X_max, selected_z, OUTPUT_DIR, suffix=SUFFIX)
'''
S_full    = (X_orig, P_orig)
S_hat_all = (X_hat, P_hat)
bandwidth = BASE_SIGMA * (1 / M) ** (1 / 3)
with torch.no_grad():
    term0   = varifold_sp(S_full,S_full,bandwidth)     # <S, S>
    term1   = varifold_sp(S_hat_all, S_hat_all,bandwidth)  # <Ŝ, Ŝ>
    term2   = varifold_sp(S_full,S_hat_all,bandwidth)  # <S, Ŝ>
    dist_sq = (term0 + term1 - 2 * term2).item()

dist = dist_sq ** 0.5 if dist_sq > 0 else float("nan")

print("──────────────────────────────────────────")
print(f"varifold distance² : {dist_sq:.6e}")
print(f"varifold distance  : {dist:.6e}")
print("──────────────────────────────────────────")