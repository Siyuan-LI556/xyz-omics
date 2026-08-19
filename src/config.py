# ── Experiment identity ────────────────────────────────
RUN_ID           = 202            # increment for each experiment
SUBSAMPLE_METHOD = "kmeans"      # "kmeans" | "random"
KERNEL_TYPE      = "isotropic"   # "isotropic" | "anisotropic"
OPTIMIZER        = "adam"       # "lbfgs" | "adam"
Input = "hat_run69_best_performence_of_real_dataset.vtp"     # "MB35_BL2_L20_11.npz" | "test_subsample.vtk" |
                                      # "D076_1L_approx200um" | "all_slices_C57BL6J.npz"

# Derived output suffix — used for all file names, do not edit manually
SUFFIX = f"run{RUN_ID:02d}_{SUBSAMPLE_METHOD}_{KERNEL_TYPE}_{OPTIMIZER}"

# ── Subsampling ────────────────────────────────────────
M = 500000            # number of representative points

# ── Varifold kernel parameters ─────────────────────────
BASE_SIGMA = 1    # isotropic bandwidth (scaled by M at runtime)
SIGMA_XY   = 0.0175   # anisotropic: bandwidth for x/y directions
SIGMA_Z    = 0.05    # anisotropic: bandwidth for z direction

# ── LBFGS optimizer ────────────────────────────────────
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 100
TOL          = 1e-6
PATIENCE     = 3

# ── Adam optimizer ─────────────────────────────────────
ADAM_LR_X    = 0.001
ADAM_LR_P    = 0.001
ADAM_EPOCHS  = 800

# Stopping: target relative residual eps = ||S - S_hat|| / ||S||.
ADAM_TARGET_EPS  = 0.003  # 0.05 = stop at 5% relative error
ADAM_MIN_EPOCHS  = 300    # no stall test before this
ADAM_STALL_WINDOW = 50    # stall test compares mean(last 50) vs mean(previous 50)
ADAM_STALL_TOL   = 1e-5   # minimum relative improvement between those windows
ADAM_SOFTPLUS_P  = True   # parametrize P = softplus(theta) instead of clamping at 0

FREEZE_P = True
ADAM_LR_DECAY    = True   # halve the learning rates on plateau

# ── Algorithm 3: global mini-batch (no tiling) ─────────
# Cost per optimizer step is (M^2 + b*M) kernel pairs. The exact ||mu_hat||^2 self-term
# is O(M^2) and mini-batching does NOT reduce it. Measured on this box (Quadro T2000,
# ~1.9e10 pairs/s fwd+bwd) for MB35 (N=30397307, K=ceil(N/b)=304):
#     MB_M=1e5 -> ~1.0 s/step, ~5 min/epoch   -> 200 epochs ~= 1 day
#     MB_M=1e6 -> ~57  s/step, ~4.8 h/epoch   -> 200 epochs ~= 40 days   <-- current
# M=1e6 needs block-sparse kernels (KeOps `ranges` + grid binning) to be practical.
# Keep b ~= M: that balances the M^2 self-term against the b*M cross-term.
MB_ENABLE     = False     # route run_optimizer's adam branch to the mini-batch loop
MB_M          = 500000     # representative points (must be < N; MB35 N=30397307)
MB_BATCH_SIZE = 100000     # b; K = ceil(N/b) = 304 on MB35
MB_EPOCHS     = 200
MB_EVAL_EVERY = 10        # full-batch eval + best snapshot cadence (epochs)
# Subsampling ||mu||^2 over-weights the i=j diagonal by N/MB_NORMSQ_SUB, which biases
# the reported eps (152x at 200k on MB35, 30x at 1M). 0 = exact, unaffordable at N=30M.
MB_NORMSQ_SUB = 1000000
MB_EXPORT_ORIG = False    # orig.vtp is dataset-, not run-, dependent; ~1 GB at N=30M

# ── Tiling config ──────────────────────────────────────
N_GRID      = 10            # n x n blocks

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
