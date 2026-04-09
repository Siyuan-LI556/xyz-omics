# ── Experiment identity ────────────────────────────────
RUN_ID           = 1             # increment for each experiment
SUBSAMPLE_METHOD = "kmeans"      # "kmeans" | "random"
KERNEL_TYPE      = "anisotropic" # "isotropic" | "anisotropic"

# Derived output suffix — used for all file names, do not edit manually
SUFFIX = f"run{RUN_ID:02d}_{SUBSAMPLE_METHOD}_{KERNEL_TYPE}"

# ── Subsampling ────────────────────────────────────────
M = 1000            # number of representative points

# ── Varifold kernel parameters ─────────────────────────
BASE_SIGMA = 1.0    # isotropic bandwidth (scaled by M at runtime)
SIGMA_XY   = 0.02   # anisotropic: bandwidth for x/y directions
SIGMA_Z    = 0.1    # anisotropic: bandwidth for z direction

# ── LBFGS optimizer ────────────────────────────────────
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 500
TOL          = 1e-6
PATIENCE     = 3
