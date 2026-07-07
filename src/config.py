# ── Experiment identity ────────────────────────────────
RUN_ID           = 44             # increment for each experiment
SUBSAMPLE_METHOD = "kmeans"      # "kmeans" | "random"
KERNEL_TYPE      = "isotropic"   # "isotropic" | "anisotropic"
OPTIMIZER        = "lbfgs"       # "lbfgs" | "adam"
Input = "test_subsample.vtk"
# Derived output suffix — used for all file names, do not edit manually
SUFFIX = f"run{RUN_ID:02d}_{SUBSAMPLE_METHOD}_{KERNEL_TYPE}_{OPTIMIZER}"

# ── Subsampling ────────────────────────────────────────
M = 80000            # number of representative points

# ── Varifold kernel parameters ─────────────────────────
BASE_SIGMA = 1.0    # isotropic bandwidth (scaled by M at runtime)
SIGMA_XY   = 0.1   # anisotropic: bandwidth for x/y directions
SIGMA_Z    = 0.1    # anisotropic: bandwidth for z direction

# ── LBFGS optimizer ────────────────────────────────────
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 500
TOL          = 1e-6
PATIENCE     = 3

# ── Adam optimizer ─────────────────────────────────────
ADAM_LR_X    = 0.01   # learning rate for X_hat
ADAM_LR_P    = 0.005  # learning rate for P_hat
ADAM_EPOCHS  = 5000
ADAM_TOL     = 2e-6
ADAM_PATIENCE = 50

# ── Tiling config ──────────────────────────────────────
N_GRID      = 2            # n x n blocks

TILE_MODE   = "blocks_and_strips"  # "blocks_only"  | "blocks_and_strips" | "blocks_and_overlaps"
M_TOTAL     = M            # global representative-point budget, split across regions
strip_width = 0.05
# blocks_and_strips: refine seam bands carved from the optimized block cloud.
STRIP_MOVE_P = False       # False: relocate points (X) only, keep each point's feature P
BPRIME_OPTIMIZE_OVERLAP = True
BPRIME_OVERLAP_KEEP_RATIO = 0.5  # e.g. 0.5 keeps half, 1/3 keeps one third
