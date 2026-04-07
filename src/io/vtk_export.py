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