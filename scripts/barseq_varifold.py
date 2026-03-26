# scripts/barseq_varifold.py
import os
import torch
from src.config import M, BASE_SIGMA, LR, MAX_ITER, HISTORY_SIZE, EPOCHS
from src.io.loader import load_barseq
from src.subsampling.kmeans import kmeans_subsample
from src.losses.varifold import varifold_sp
from src.optim.LBFGS import optimize_lbfgs
from src.io.vtk_export import export_vtp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "D076_1L_approx200um.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Derived parameters ─────────────────────────────────
bandwidth_varifold = BASE_SIGMA * (1 / M) ** (1 / 3)
print(f"bandwidth_varifold set to {bandwidth_varifold:.4f} based on M={M} representative points.")

# ── Load data ──────────────────────────────────────────
X_orig, P_orig, X_min, X_max = load_barseq(DATA_FILE, device)
S = (X_orig, P_orig)

# ── Subsample ──────────────────────────────────────────
print(f"Preparing to compress {X_orig.shape[0]} brain cell points into {M} representative points...")
X_hat, P_hat = kmeans_subsample(X_orig, P_orig, M, device)

# ── Optimize ───────────────────────────────────────────
X_hat, P_hat, loss_history = optimize_lbfgs(
    S, X_hat, P_hat, bandwidth_varifold,
    lr=LR, max_iter=MAX_ITER, history_size=HISTORY_SIZE, epochs=EPOCHS
)

# ── Export ─────────────────────────────────────────────
export_vtp(X_orig, P_orig, X_hat, P_hat, X_min, X_max, OUTPUT_DIR)