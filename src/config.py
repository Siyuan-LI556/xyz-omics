# ── Experiment identity ────────────────────────────────
RUN_ID           = 94            # increment for each experiment
SUBSAMPLE_METHOD = "kmeans"      # "kmeans" | "random"
KERNEL_TYPE      = "isotropic"   # "isotropic" | "anisotropic"
OPTIMIZER        = "adam"       # "lbfgs" | "adam"
Input = "test_subsample.vtk"     # "MB35_BL2_L20_11.npz" | "test_subsample.vtk"

# Derived output suffix — used for all file names, do not edit manually
SUFFIX = f"run{RUN_ID:02d}_{SUBSAMPLE_METHOD}_{KERNEL_TYPE}_{OPTIMIZER}"

# ── Subsampling ────────────────────────────────────────
M = 50000            # number of representative points

# ── Varifold kernel parameters ─────────────────────────
BASE_SIGMA = 0.25    # isotropic bandwidth (scaled by M at runtime)
SIGMA_XY   = 0.003   # anisotropic: bandwidth for x/y directions
SIGMA_Z    = 0.003    # anisotropic: bandwidth for z direction

# ── LBFGS optimizer ────────────────────────────────────
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 500
TOL          = 1e-6
PATIENCE     = 3

# ── Adam optimizer ─────────────────────────────────────
ADAM_LR_X    = 0.0014
ADAM_LR_P    = 0.001
ADAM_EPOCHS  = 500

# Stopping: target relative residual eps = ||S - S_hat|| / ||S||.
ADAM_TARGET_EPS  = 0.005  # 0.05 = stop at 5% relative error
ADAM_MIN_EPOCHS  = 300    # no stall test before this
ADAM_STALL_WINDOW = 50    # stall test compares mean(last 50) vs mean(previous 50)
ADAM_STALL_TOL   = 1e-4   # minimum relative improvement between those windows
ADAM_SOFTPLUS_P  = True   # parametrize P = softplus(theta) instead of clamping at 0

FREEZE_P = True
ADAM_LR_DECAY    = True   # halve the learning rates on plateau

# ── Tiling config ──────────────────────────────────────
N_GRID      = 3            # n x n blocks

TILE_MODE   = "blocks_only"  # "blocks_only" | "blocks_and_strips" | "blocks_and_overlaps" | "blocks_expanded"
M_TOTAL     = M            # global representative-point budget, split across regions
strip_width = 0.15
# blocks_and_strips: refine seam bands carved from the optimized block cloud.
STRIP_MOVE_P = False       # False: relocate points (X) only, keep each point's feature P
# blocks_expanded: per-side boundary halo (normalized). Optimize block+halo, slice back to
# the true block bounds. Pick ~2–3× kernel bandwidth so the seam density stays continuous.
EXPAND_HALO = 0.02
BPRIME_OPTIMIZE_OVERLAP = True
BPRIME_OVERLAP_KEEP_RATIO = 0.5  # e.g. 0.5 keeps half, 1/3 keeps one third
