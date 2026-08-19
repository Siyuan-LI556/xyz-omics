# scripts/barseq_varifold.py
import os
import torch
from src.config import (
    RUN_ID, SUBSAMPLE_METHOD, KERNEL_TYPE, OPTIMIZER, SUFFIX, Input,
    M, BASE_SIGMA, SIGMA_XY, SIGMA_Z, MB_ENABLE,
)
from src.io.loader import load_barseq, load_middle_slices
from src.subsampling import subsample
from src.optim import run_optimizer
from src.losses.varifold import make_varifold_fn, isotropic_sigma
from src.io.vtk_export import export_orig_vtp, export_hat_vtp, export_middle_slices_vtp, export_hat_middle_slices_vtp
from src.io.plot import plot_loss_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run config  : {SUFFIX}")

# Seed everything from RUN_ID: the random init is otherwise unseeded, so repeated
# draws (Section 5.5.1) are obtained by changing RUN_ID and stay reproducible.
torch.manual_seed(RUN_ID)

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", Input)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(DATA_FILE, device)
S = (X_orig, P_orig)
DIM = X_orig.shape[1]

# ── Subsample ──────────────────────────────────────────
print(f"Subsampling method: {SUBSAMPLE_METHOD}  →  {M} representative points")
X_hat, P_hat = subsample(SUBSAMPLE_METHOD, X_orig, P_orig, M, device)

# ── Build varifold function ────────────────────────────
print(f"Kernel type: {KERNEL_TYPE}")
if KERNEL_TYPE == "isotropic":
    print(f"Isotropic bandwidth: {isotropic_sigma(BASE_SIGMA, M, DIM):.5f} (dim={DIM})")
elif KERNEL_TYPE == "anisotropic":
    print(f"Anisotropic bandwidth: sigma_xy={SIGMA_XY}, sigma_z={SIGMA_Z}")
varifold_fn = make_varifold_fn(KERNEL_TYPE, M, BASE_SIGMA, SIGMA_XY, SIGMA_Z, dim=DIM)

# ── Optimize ───────────────────────────────────────────
print(f"Optimizer: {OPTIMIZER}")
X_hat, P_hat, history, time_history = run_optimizer(S, X_hat, P_hat, varifold_fn)
# The mini-batch branch reports eps(t), not loss(t) — see run_optimizer's docstring.
# Keying them apart keeps plot_eps_comparison from re-deriving eps from an eps curve.
history_key = "eps_history" if MB_ENABLE else "loss_history"

# ── Export ─────────────────────────────────────────────
export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
export_hat_vtp(X_hat, P_hat, X_min, X_max, OUTPUT_DIR, suffix=SUFFIX)
# ── Final metric + run history (for the eps comparison plots) ──
S_full    = (X_orig, P_orig)
S_hat_all = (X_hat, P_hat)
with torch.no_grad():
    term0   = varifold_fn(S_full, S_full)          # <S, S> = ||mu||^2
    term1   = varifold_fn(S_hat_all, S_hat_all)    # <Ŝ, Ŝ>
    term2   = varifold_fn(S_full, S_hat_all)       # <S, Ŝ>
    dist_sq = (term0 + term1 - 2 * term2).item()

norm_sq = term0.item()
eps = (max(dist_sq, 0.0) / norm_sq) ** 0.5 if norm_sq > 0 else float("nan")

print("──────────────────────────────────────────")
print(f"varifold distance² : {dist_sq:.6e}")
print(f"relative residual ε: {eps:.4%}")
print("──────────────────────────────────────────")

# Small results record — loss_history is the full ||mu - mu_hat||^2, so the eps(t)
# curve is recovered later as sqrt(loss / norm_sq) by plot_eps_comparison.
torch.save({
    "suffix": SUFFIX, "run_id": RUN_ID, "M": M, "N": X_orig.shape[0],
    "method": SUBSAMPLE_METHOD, "optimizer": OPTIMIZER, "kernel": KERNEL_TYPE,
    history_key: history, "time_history": time_history,
    "norm_sq": norm_sq, "dist_sq": dist_sq, "eps": eps,
}, os.path.join(OUTPUT_DIR, f"results_{SUFFIX}.pt"))

#plot_loss_curve(history, time_history, output_dir=OUTPUT_DIR, suffix=SUFFIX)

# ── Middle 3 slices comparison ─────────────────────────
#X_mid, P_mid, selected_z, slice_id = load_middle_slices(DATA_FILE, n=3)
#export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, OUTPUT_DIR)
#export_hat_middle_slices_vtp(X_hat, P_hat, X_min, X_max, selected_z, OUTPUT_DIR, suffix=SUFFIX)