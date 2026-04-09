# src/io/vtk_export.py
import os
import numpy as np
import pyvista as pv


def export_orig_vtp(X_orig, P_orig, X_min, X_max, output_dir):
    """
    Export original point cloud to ParaView format (.vtp).
    """
    X_orig_restored = (X_orig.detach() * (X_max - X_min) + X_min).cpu().numpy()
    p_orig_final    = P_orig.detach().cpu().numpy()

    cloud_orig = pv.PolyData(X_orig_restored)

    cloud_orig.point_data["Gene_weight"] = p_orig_final

    file_orig = os.path.join(output_dir, "orig.vtp")
    cloud_orig.save(file_orig)
    print(f"Saved original point cloud to {file_orig}")


def export_hat_vtp(X_hat, P_hat, X_min, X_max, output_dir, suffix="kmeans"):
    """
    Export representative point cloud to ParaView format (.vtp).
    """
    X_hat_restored = (X_hat.detach() * (X_max - X_min) + X_min).cpu().numpy()
    p_hat_final    = P_hat.detach().cpu().numpy()

    cloud_hat = pv.PolyData(X_hat_restored)

    cloud_hat.point_data["Gene_weight"] = p_hat_final

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
    z_step = selected_z[1] - selected_z[0]  # step between adjacent slices (e.g. 0.2)
    tol = z_step / 2

    mask = (z_hat >= selected_z[0] - tol) & (z_hat <= selected_z[-1] + tol)
    n_kept = mask.sum()
    print(f"Hat points in middle z-range [{selected_z[0]:.2f}, {selected_z[-1]:.2f}]: {n_kept}")

    X_filtered = X_hat_restored[mask].copy()
    P_filtered = P_hat_np[mask]

    X_filtered[:, 2] = 0.1  # flatten to same plane as original export

    cloud = pv.PolyData(X_filtered)
    cloud.point_data["Gene_weight"]      = P_filtered
    cloud.point_data["Total_expression"] = P_filtered.sum(axis=1)

    file_path = os.path.join(output_dir, f"hat_middle_slices_{suffix}.vtp")
    cloud.save(file_path)
    print(f"Saved compressed middle slices to {file_path}")


def export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, output_dir):
    """
    Export the merged middle slices and each individual slice as .vtp for ParaView.
    """
    X_np = X_mid.copy()
    P_np = P_mid

    # Flatten all points onto z=0 so the 3 merged slices appear as one plane.
    X_np[:, 2] = 0.1

    # Merged file
    cloud = pv.PolyData(X_np)
    cloud.point_data["Gene_weight"]      = P_np
    cloud.point_data["Total_expression"] = P_np.sum(axis=1)

    merged_file = os.path.join(output_dir, "middle_slices_orig.vtp")
    cloud.save(merged_file)
    print(f"Saved merged middle slices to {merged_file}")