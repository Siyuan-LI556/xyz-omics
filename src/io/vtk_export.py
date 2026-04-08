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


def export_middle_slices_vtp(X_mid, P_mid, selected_z, slice_id, output_dir):
    """
    Export the merged middle slices and each individual slice as .vtp for ParaView.
    """
    X_np = X_mid.copy()
    P_np = P_mid

    # Flatten all points onto z=0 so the 3 merged slices appear as one plane.
    X_np[:, 2] = 0.0

    # Merged file
    cloud = pv.PolyData(X_np)
    cloud.point_data["Gene_weight"]      = P_np
    cloud.point_data["Total_expression"] = P_np.sum(axis=1)

    merged_file = os.path.join(output_dir, "middle_slices_merged.vtp")
    cloud.save(merged_file)
    print(f"Saved merged middle slices to {merged_file}")