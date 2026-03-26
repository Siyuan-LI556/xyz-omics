# Subsampling parameters
M = 1000            # Number of representative points

# Varifold kernel parameters
BASE_SIGMA = 1.0    # Base bandwidth for varifold kernel, can be tuned based on data scale and desired smoothness

# L-BFGS optimizer parameters
LR           = 0.1
MAX_ITER     = 20
HISTORY_SIZE = 10
EPOCHS       = 50
TOL = 1e-6
PATIENCE = 5