# src/subsampling/kmeans.py
import torch
from sklearn.cluster import KMeans
from pykeops.torch import LazyTensor
'''
def kmeans_subsample(X: torch.Tensor, P: torch.Tensor, M: int, device: torch.device):
    """
    Subsample by K-means to better capture the spatial distribution and gene expression
    diversity of the original dataset. Random sampling may not be sufficient to
    preserve important biological patterns in the data.
    """
    kmeans = KMeans(n_clusters=M, random_state=0)
    labels = kmeans.fit_predict(X.cpu().numpy())
    centers = torch.tensor(kmeans.cluster_centers_, device=device)

    P_hat = torch.zeros((M, P.shape[1]), device=device)

    for i in range(M):
        mask = torch.tensor(labels == i, device=device)
        if mask.sum() > 0:
            P_hat[i] = P[mask].mean(dim=0)

    X_hat = centers.clone().requires_grad_(True)
    P_hat = P_hat.clone().requires_grad_(True)

    return X_hat, P_hat
'''

def kmeans_subsample(X: torch.Tensor, P: torch.Tensor, M: int,
                     device: torch.device, n_iter: int = 10, seed: int = 0):
    """
    GPU K-means (KeOps). Cluster on spatial coords X, then take per-cluster
    mean of P as the reduced features. Fully on-GPU, no N x M matrix
    materialized, P aggregation vectorized via scatter_add.
    Returns (X_hat, P_hat) as leaf tensors with requires_grad=True.
    """
    X = X.to(device).contiguous()
    P = P.to(device).contiguous()
    N, D = X.shape
    Dp = P.shape[1]
    M = min(int(M), N)                        # cannot have more clusters than points

    # Init centroids: M random points from the data
    g = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=device)[:M]
    c = X[perm].clone()                       # (M, D)

    x_i = LazyTensor(X.view(N, 1, D))         # (N, 1, D) symbolic

    # Lloyd iterations
    for _ in range(n_iter):
        c_j = LazyTensor(c.view(1, M, D))     # (1, M, D)
        # (N, M) symbolic squared dist -> nearest centroid; matrix never stored
        cl = ((x_i - c_j) ** 2).sum(-1).argmin(dim=1).view(-1).long()  # (N,)

        # Update centroids: per-cluster sum / count, vectorized
        c_sum = torch.zeros(M, D, device=device, dtype=X.dtype)
        c_sum.scatter_add_(0, cl[:, None].expand(N, D), X)
        counts = torch.bincount(cl, minlength=M).view(M, 1)
        nonempty = counts.view(-1) > 0
        c_new = c.clone()
        c_new[nonempty] = c_sum[nonempty] / counts[nonempty].to(X.dtype)
        c = c_new                             # keep old centroid for empty clusters

    # Reassign with final centroids so X_hat / P_hat share the same labels
    c_j = LazyTensor(c.view(1, M, D))
    cl = ((x_i - c_j) ** 2).sum(-1).argmin(dim=1).view(-1).long()

    # Per-cluster mean of P (replaces the old for-loop)
    P_sum = torch.zeros(M, Dp, device=device, dtype=P.dtype)
    P_sum.scatter_add_(0, cl[:, None].expand(N, Dp), P)
    counts = torch.bincount(cl, minlength=M).clamp_min_(1).view(M, 1).to(P.dtype)
    P_hat = P_sum / counts

    X_hat = c.detach().clone().requires_grad_(True)
    P_hat = P_hat.detach().clone().requires_grad_(True)
    return X_hat, P_hat
