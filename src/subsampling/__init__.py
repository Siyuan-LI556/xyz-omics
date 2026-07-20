from src import config
from .kmeans import kmeans_subsample
from .random import random_subsample


def subsample(method, X, P, M, device):
    """Dispatch to the configured subsampling method. Returns (X_hat, P_hat)."""
    if method == "kmeans":
        X_hat, P_hat = kmeans_subsample(X, P, M, device)
    elif method == "random":
        X_hat, P_hat = random_subsample(X, P, M)
    else:
        raise ValueError(f"Unknown SUBSAMPLE_METHOD: '{method}'.")

    if config.FREEZE_P:
        P_hat.requires_grad_(False)
    return X_hat, P_hat
