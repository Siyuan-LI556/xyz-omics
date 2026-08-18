# scripts/barseq_varifold_minibatch.py
# Algorithm 3: one global representative measure optimized by mini-batch Adam.
# No tiling. Reuses the existing loader / kmeans init / varifold factory / exporters.
import os
import torch
from src.config import (
    SUBSAMPLE_METHOD, KERNEL_TYPE, SUFFIX, Input,
    BASE_SIGMA, SIGMA_XY, SIGMA_Z, FREEZE_P,
    ADAM_LR_X, ADAM_LR_P, ADAM_TARGET_EPS, ADAM_SOFTPLUS_P,
    MB_M, MB_BATCH_SIZE, MB_EPOCHS, MB_EVAL_EVERY, MB_NORMSQ_SUB,
)
from src.io.loader import load_barseq
from src.subsampling import subsample
from src.losses.varifold import make_varifold_fn
from src.optim.Adam import optimize_adam_minibatch
from src.io.vtk_export import export_orig_vtp, export_hat_vtp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run config  : {SUFFIX}_mb")

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", Input)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load (whole cloud resident on device) ──────────────
X_orig, P_orig, W_orig, P_norm_orig, X_min, X_max = load_barseq(DATA_FILE, device)
S_full = (X_orig, P_orig)
N = X_orig.shape[0]
if MB_M >= N:
    raise ValueError(f"MB_M ({MB_M}) must be < N ({N}); otherwise there is no reduction.")
print(f"N={N}  MB_M={MB_M}  b={MB_BATCH_SIZE}  K={-(-N // MB_BATCH_SIZE)}")

# ── Init one global representative measure (kmeans) ────
X_hat, P_hat = subsample(SUBSAMPLE_METHOD, X_orig, P_orig, MB_M, device)

# ── Varifold (isotropic bandwidth scaled by MB_M) ──────
print(f"Kernel type: {KERNEL_TYPE}")
if KERNEL_TYPE == "isotropic":
    print(f"Isotropic bandwidth: {BASE_SIGMA * (1 / MB_M) ** (1 / 3):.4f}")
varifold_fn = make_varifold_fn(KERNEL_TYPE, MB_M, BASE_SIGMA, SIGMA_XY, SIGMA_Z)

# ── Optimize (mini-batch Adam) ─────────────────────────
X_hat, P_hat = optimize_adam_minibatch(
    S_full, X_hat, P_hat, varifold_fn,
    batch_size=MB_BATCH_SIZE, epochs=MB_EPOCHS,
    lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
    eval_every=MB_EVAL_EVERY, normsq_sub=MB_NORMSQ_SUB,
    target_eps=ADAM_TARGET_EPS, softplus_P=ADAM_SOFTPLUS_P,
    freeze_P=FREEZE_P,
)

# ── Export ─────────────────────────────────────────────
export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
export_hat_vtp(X_hat, P_hat, X_min, X_max, OUTPUT_DIR, suffix=SUFFIX + "_mb")
