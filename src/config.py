# Subsampling parameters
M = 1000            # Number of representative points

# Varifold kernel parameters
BASE_SIGMA = 1.0    # Base bandwidth for varifold kernel, can be tuned based on data scale and desired smoothness

# LBFGS optimizer parameters
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 500
TOL = 1e-6
PATIENCE = 3

# Anisotropic varifold kernel parameters
SIGMA_XY = 0.02   # bandwidth for x and y directions
SIGMA_Z  = 0.1   # bandwidth for z direction (slice thickness)