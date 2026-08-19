# src/io/loader.py
import numpy as np
import torch
import pyvista as pv


def load_barseq(filepath: str, device: torch.device, return_pnorm: bool = True):
    """
    Load BARSeq data from .npz/.vtk/.vtp file and normalize spatial coordinates to [0, 1] range.
    Returns tensors and normalization parameters for later restoration.

    return_pnorm=False skips building P_norm (an extra N x d tensor on `device`) and
    returns None in its place. Callers that only need the raw measure (X, P) should pass
    False: on the 30M-point clouds P_norm alone is ~0.4 GB of otherwise dead VRAM.
    """

    if filepath.endswith(".npz"):
        print("Loading .npz file...")
        data = np.load(filepath)

        # Barseq data contains 62,453 points with 39 gene features each
        Position_BarSeq = data["X"]       # a.k.a. x_i in xIV-LDDMM paper
        Feature_BarSeq  = data["nu_X"]    # a.k.a. w_i*p_i in xIV-LDDMM paper

    elif filepath.endswith(".vtk") or filepath.endswith(".vtp"):
        print("Loading .vtk/.vtp file...")
        mesh = pv.read(filepath)

        Position_BarSeq = mesh.points

        if "nu_X" in mesh.point_data:
            Feature_BarSeq = mesh.point_data["nu_X"]
        else:
            raise KeyError(
                f"Cannot find 'nu_X' in VTK point_data. "
                f"Available keys: {list(mesh.point_data.keys())}"
            )

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    Position_BarSeq = np.asarray(Position_BarSeq)
    Feature_BarSeq = np.asarray(Feature_BarSeq)

    print(
        f"Loaded data with {Position_BarSeq.shape[0]} points "
        f"and {Feature_BarSeq.shape[1]} gene features."
    )
    print(f"Feature range is {Feature_BarSeq.min()}, {Feature_BarSeq.max()}")

    # Convert to PyTorch tensors
    X = torch.tensor(Position_BarSeq, dtype=torch.float32, device=device)
    P = torch.tensor(Feature_BarSeq, dtype=torch.float32, device=device)

    # Decompose nu_X = w_i * p_i
    # w_i : scalar total gene expression per cell, L1 sum over all genes
    # p_i : normalized gene profile, probability vector, sums to 1
    W = P.sum(dim=1)                       # shape (N,)
    P_norm = P / (W.unsqueeze(1) + 1e-12) if return_pnorm else None  # shape (N, d)

    print(
        f"w_i  — min={W.min().item():.4f}, "
        f"max={W.max().item():.4f}, "
        f"mean={W.mean().item():.4f}"
    )

    # Normalize spatial coordinates to [0, 1] range
    X_min, _ = torch.min(X, dim=0)
    X_max, _ = torch.max(X, dim=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X = (X - X_min) / X_range

    return X, P, W, P_norm, X_min, X_max


def load_middle_slices(filepath: str, n: int = 3):
    """
    Load BARSeq data and return the middle n slices along the z-axis.
    Supports .npz/.vtk/.vtp file.
    """

    if filepath.endswith(".npz"):
        print("Loading .npz file for slice extraction...")
        data = np.load(filepath)

        X = data["X"]
        P = data["nu_X"]

    elif filepath.endswith(".vtk") or filepath.endswith(".vtp"):
        print("Loading .vtk/.vtp file for slice extraction...")
        mesh = pv.read(filepath)

        X = mesh.points

        if "nu_X" in mesh.point_data:
            P = mesh.point_data["nu_X"]
        else:
            raise KeyError(
                f"Cannot find 'nu_X' in VTK point_data. "
                f"Available keys: {list(mesh.point_data.keys())}"
            )

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    if X.shape[1] < 3:
        raise ValueError(
            f"Cannot extract z-slices because X has no z-axis. X shape is {X.shape}"
        )

    z_vals = X[:, 2]

    # Round z to the nearest integer to handle floating-point noise across
    # points that nominally belong to the same physical slice.
    z_rounded = np.round(z_vals, decimals=1)
    unique_z = np.unique(z_rounded)

    n_slices = len(unique_z)
    print(f"Number of slices {n_slices}")

    mid = n_slices // 2
    half = n // 2

    selected_z = unique_z[mid - half: mid - half + n]
    print(f"Selected middle {n} z-slices: {selected_z}")

    mask = np.isin(z_rounded, selected_z)

    X_mid = X[mask]
    P_mid = P[mask]

    slice_id = np.searchsorted(selected_z, z_rounded[mask]).astype(np.int32)

    return X_mid, P_mid, selected_z, slice_id