# scripts/barseq_varifold_minibatch.py
# Algorithm 3: one global representative measure optimized by mini-batch Adam.
# No tiling. Reuses the existing loader / kmeans init / varifold factory / exporters.
import os
import torch
from src.config import (
    RUN_ID, SUBSAMPLE_METHOD, KERNEL_TYPE, SUFFIX, Input,
    BASE_SIGMA, SIGMA_XY, SIGMA_Z, FREEZE_P,
    ADAM_LR_X, ADAM_LR_P, ADAM_TARGET_EPS, ADAM_SOFTPLUS_P,
    MB_M, MB_BATCH_SIZE, MB_EPOCHS, MB_EVAL_EVERY, MB_NORMSQ_SUB, MB_EXPORT_ORIG,
)
from src.io.loader import load_barseq
from src.subsampling import subsample
from src.losses.varifold import make_varifold_fn, isotropic_sigma
from src.optim.Adam import optimize_adam_minibatch
from src.io.vtk_export import export_orig_vtp, export_hat_vtp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run config  : {SUFFIX}_mb")

torch.manual_seed(RUN_ID)

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", Input)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load (whole cloud resident on device) ──────────────
# return_pnorm=False: this pipeline only needs the raw measure (X, P). Building P_norm
# costs another N x d tensor on the GPU (~0.4 GB at N=30M) that nothing here reads.
X_orig, P_orig, W_orig, _, X_min, X_max = load_barseq(DATA_FILE, device, return_pnorm=False)
S_full = (X_orig, P_orig)
N, DIM = X_orig.shape
if MB_M >= N:
    raise ValueError(f"MB_M ({MB_M}) must be < N ({N}); otherwise there is no reduction.")
print(f"N={N}  dim={DIM}  MB_M={MB_M}  b={MB_BATCH_SIZE}  K={-(-N // MB_BATCH_SIZE)}")

# ── Init one global representative measure (kmeans) ────
X_hat, P_hat = subsample(SUBSAMPLE_METHOD, X_orig, P_orig, MB_M, device)

# ── Varifold (bandwidth scaled by MB_M in the data's own dimension) ──
print(f"Kernel type: {KERNEL_TYPE}")
if KERNEL_TYPE == "isotropic":
    print(f"Isotropic bandwidth: {isotropic_sigma(BASE_SIGMA, MB_M, DIM):.5f}  "
          f"(= BASE_SIGMA * MB_M^(-1/{DIM}))")
varifold_fn = make_varifold_fn(KERNEL_TYPE, MB_M, BASE_SIGMA, SIGMA_XY, SIGMA_Z, dim=DIM)

# ── Optimize (mini-batch Adam) ─────────────────────────
X_hat, P_hat, eps_history, time_history = optimize_adam_minibatch(
    S_full, X_hat, P_hat, varifold_fn,
    batch_size=MB_BATCH_SIZE, epochs=MB_EPOCHS,
    lr_X=ADAM_LR_X, lr_P=ADAM_LR_P,
    eval_every=MB_EVAL_EVERY, normsq_sub=MB_NORMSQ_SUB,
    target_eps=ADAM_TARGET_EPS, softplus_P=ADAM_SOFTPLUS_P,
    freeze_P=FREEZE_P,
)

# ── Export ─────────────────────────────────────────────
# The original cloud is dataset-, not run-, dependent and writes a ~1 GB .vtp at
# N=30M (plus a full host copy). Off by default; flip MB_EXPORT_ORIG once per dataset.
if MB_EXPORT_ORIG:
    export_orig_vtp(X_orig, P_orig, X_min, X_max, OUTPUT_DIR)
else:
    print(f"Skipping orig.vtp export ({N} points) — set MB_EXPORT_ORIG=True to write it.")
export_hat_vtp(X_hat, P_hat, X_min, X_max, OUTPUT_DIR, suffix=SUFFIX + "_mb")

# ── Run history (eps sampled at the eval cadence, for the comparison plots) ──
torch.save({
    "suffix": SUFFIX + "_mb", "run_id": RUN_ID, "M": MB_M, "N": N, "dim": DIM,
    "method": SUBSAMPLE_METHOD, "optimizer": "adam_minibatch", "kernel": KERNEL_TYPE,
    "batch_size": MB_BATCH_SIZE, "epochs": MB_EPOCHS, "eval_every": MB_EVAL_EVERY,
    "eps_history": eps_history, "time_history": time_history,
    "eps": min(eps_history) if eps_history else float("nan"),
}, os.path.join(OUTPUT_DIR, f"results_{SUFFIX}_mb.pt"))
print(f"Saved run record to {os.path.join(OUTPUT_DIR, f'results_{SUFFIX}_mb.pt')}")
