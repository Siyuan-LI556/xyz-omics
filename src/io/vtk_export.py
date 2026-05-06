# src/io/vtk_export.py
import os
import numpy as np
import pyvista as pv


def _decompose(P_np):
    """Split P = w_i * p_i into scalar weight w_i and normalized profile p_i."""
    w = P_np.sum(axis=1)                          # (N,)
    p = P_np / (w[:,None] + 1e-12)              # (N, 39)
    return w, p

def export_orig_vtp(X_orig, P_orig, X_min, X_max, output_dir):
    """
    Export original point cloud to ParaView format (.vtp).
    """
    X_orig_restored = (X_orig.detach() * (X_max - X_min) + X_min).cpu().numpy()
    P_np = P_orig.detach().cpu().numpy()
    w, p = _decompose(P_np)

    cloud_orig = pv.PolyData(X_orig_restored)
    cloud_orig.point_data["nu_X"]          = P_np   # raw w_i * p_i  (39-dim vector)
    cloud_orig.point_data["wi"]            = w ** 1/3      # scalar total expression
    cloud_orig.point_data["pi"]            = p       # normalized gene profile (39-dim)

    file_orig = os.path.join(output_dir, "orig.vtp")
    cloud_orig.save(file_orig)
    print(f"Saved original point cloud to {file_orig}")


def export_hat_vtp(X_hat, P_hat, X_min, X_max, output_dir, suffix="kmeans"):
    """
    Export representative point cloud to ParaView format (.vtp).
    """
    X_hat_restored = (X_hat.detach() * (X_max - X_min) + X_min).cpu().numpy()
    P_np = P_hat.detach().cpu().numpy()
    w, p = _decompose(P_np)

    cloud_hat = pv.PolyData(X_hat_restored)
    cloud_hat.point_data["nu_X"]          = P_np
    cloud_hat.point_data["wi"]            = w
    cloud_hat.point_data["pi"]            = p

    file_hat = os.path.join(output_dir, f"hat_{suffix}.vtp")
    cloud_hat.save(file_hat)
    print(f"Saved representative point cloud to {file_hat}")


def export_hat_middle_slices_vtp(X_hat, P_hat, X_min, X_max, selected_z, output_dir, suffix="hat"):
    """
    Export compressed (hat) points that fall within the middle z-slices range.
    """
    X_hat_restored = (X_hat.detach() * (X_max - X_min) + X_min).cpu().numpy()
    P_hat_np = P_hat.detach().cpu().numpy()

    z_hat = X_hat_restored[:, 2]
    z_step = selected_z[1] - selected_z[0]
    tol = z_step / 2

    mask = (z_hat >= selected_z[0] - tol) & (z_hat <= selected_z[-1] + tol)
    n_kept = mask.sum()
    print(f"Hat points in middle z-range [{selected_z[0]:.2f}, {selected_z[-1]:.2f}]: {n_kept}")

    X_filtered = X_hat_restored[mask].copy()
    P_filtered = P_hat_np[mask]
    w, p = _decompose(P_filtered)

    X_filtered[:, 2] = 0.1

    cloud = pv.PolyData(X_filtered)
    cloud.point_data["nu_X"]          = P_filtered
    cloud.point_data["wi"]            = w
    cloud.point_data["pi"]            = p

    file_path = os.path.join(output_dir, f"hat_middle_slices_{suffix}.vtp")
    cloud.save(file_path)
    print(f"Saved compressed middle slices to {file_path}")


def export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, output_dir):
    """
    Export the merged middle slices and each individual slice as .vtp for ParaView.
    """
    X_np = X_mid.copy()
    P_np = P_mid
    w, p = _decompose(P_np)

    X_np[:, 2] = 0.1

    cloud = pv.PolyData(X_np)
    cloud.point_data["nu_X"]          = P_np
    cloud.point_data["wi"]            = w ** 1/3
    cloud.point_data["pi"]            = p

    merged_file = os.path.join(output_dir, "middle_slices_orig.vtp")
    cloud.save(merged_file)
    print(f"Saved merged middle slices to {merged_file}")
