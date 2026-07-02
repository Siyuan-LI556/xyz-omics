# src/subsampling/random.py
import torch


def random_subsample(X: torch.Tensor, P: torch.Tensor, M: int):
    """
    Subsample random indices from the original dataset.
    """
    N = X.shape[0]
    indices = torch.randperm(N)[:M]

    X_hat = X[indices].clone().requires_grad_(True)
    P_hat = P[indices].clone().requires_grad_(True)

    return X_hat, P_hat