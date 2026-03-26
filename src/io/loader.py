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