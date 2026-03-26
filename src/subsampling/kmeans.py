# src/subsampling/kmeans.py
import torch
from sklearn.cluster import KMeans


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