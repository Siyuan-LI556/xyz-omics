# src/io/vtk_export.py
import os
import torch
import numpy as np
import pyvista as pv


def export_vtp(X_orig, P_orig, X_hat, P_hat, X_min, X_max, output_dir):
    """
    Export original and representative point clouds to ParaView format (.vtp).
    """
    print("\nOptimization complete! Exporting ParaView format files...")

    # Restore original spatial coordinates from normalized [0, 1] range
    X_hat_restored  = X_hat.detach() * (X_max - X_min) + X_min
    X_orig_restored = X_orig.detach() * (X_max - X_min) + X_min

    x_orig_final = X_orig_restored.cpu().numpy()
    x_hat_final  = X_hat_restored.cpu().numpy()
    p_orig_final = P_orig.detach().cpu().numpy()
    p_hat_final  = P_hat.detach().cpu().numpy()

    cloud_orig = pv.PolyData(x_orig_final)
    cloud_hat  = pv.PolyData(x_hat_final)

    # Fix export: export all feature and/or new summary feature for visualization to check quality
    cloud_orig.point_data["Gene_weight"] = p_orig_final
    cloud_hat.point_data["Gene_weight"]  = p_hat_final

    file_orig = os.path.join(output_dir, "orig_kmeans1.0.vtp")
    file_hat  = os.path.join(output_dir, "hat_kmeans1.0.vtp")

    cloud_orig.save(file_orig)
    cloud_hat.save(file_hat)

    print(f"Saved original point cloud to {file_orig}")
    print(f"Saved representative point cloud to {file_hat}")