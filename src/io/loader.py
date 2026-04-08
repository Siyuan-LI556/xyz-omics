# src/io/loader.py
import numpy as np
import torch


def load_barseq(filepath: str, device: torch.device):
    """
    Load BARSeq data from .npz file and normalize spatial coordinates to [0, 1] range.
    Returns tensors and normalization parameters for later restoration.
    """
    print("Loading .npz file...")
    data = np.load(filepath)

    # Barseq data contains 62,453 points with 39 gene features each
    Position_BarSeq = data['X']      # a.k.a. x_i in xIV-LDDMM paper
    Feature_BarSeq  = data['nu_X']   # a.k.a. w_i*p_i in xIV-LDDMM paper, total gene expression per cell

    print(f"Loaded data with {Position_BarSeq.shape[0]} points and {Feature_BarSeq.shape[1]} gene features.")
    print(f"Feature range is {Feature_BarSeq.min()}, {Feature_BarSeq.max()}")

    # Convert to PyTorch tensors
    X = torch.tensor(Position_BarSeq, dtype=torch.float32, device=device)
    P = torch.tensor(Feature_BarSeq,  dtype=torch.float32, device=device)

    # Normalize spatial coordinates to [0, 1] range
    X_min, _ = torch.min(X, dim=0)
    X_max, _ = torch.max(X, dim=0)
    X = (X - X_min) / (X_max - X_min)

    return X, P, X_min, X_max


def load_middle_slices(filepath: str, n: int = 3):
    """
    Load BARSeq data and return the middle n slices along the z-axis (original coordinates).
    """
    print("Loading .npz file for slice extraction...")
    data = np.load(filepath)
    X = data['X']
    P = data['nu_X']
    z_vals = X[:, 2]
    # Round z to the nearest integer to handle floating-point noise across
    # points that nominally belong to the same physical slice.
    #print(f"Raw z unique values: {np.unique(X[:, 2])}")
    z_rounded = np.round(z_vals, decimals=1)
    unique_z = np.unique(z_rounded)
    n_slices = len(unique_z)
    print(f"Number of slices {n_slices}")

    mid = n_slices // 2
    half = n // 2
    selected_z = unique_z[mid - half : mid - half + n]
    print(f"Selected middle {n} z-slices: {selected_z}")

    mask = np.isin(z_rounded, selected_z)
    X_mid = X[mask]
    P_mid = P[mask]
    slice_id = np.searchsorted(selected_z, z_rounded[mask]).astype(np.int32)

    return X_mid, P_mid, selected_z, slice_id