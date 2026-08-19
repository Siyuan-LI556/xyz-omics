from pykeops.torch import LazyTensor
import torch

# %% Declare kernel functions and varifold loss
def gaussian_kernel(x_i, x_j, sigma):
    dist_sq = ((x_i - x_j) ** 2).sum(-1)
    return (-dist_sq / (2 * sigma ** 2)).exp()

# function of gaussian anisotropic
def gaussian_kernel_anisotropic(x_i, x_j, sigma_xy, sigma_z):
    """
    Anisotropic Gaussian kernel with different bandwidths for xy-plane and z-axis.
    Diagonal covariance: diag(sigma_xy^2, sigma_xy^2, sigma_z^2)
    """
    diff   = x_i - x_j
    diff_x = diff[:, :, 0]   # x component (LazyTensor slices last dim only, no ellipsis)
    diff_y = diff[:, :, 1]   # y component
    diff_z = diff[:, :, 2]   # z component

    dist_sq = (diff_x ** 2 + diff_y ** 2) / (2 * sigma_xy ** 2) + \
              (diff_z ** 2)               / (2 * sigma_z  ** 2)

    #Sig = LazyTensor(torch.cat([sigma_xy, sigma_xy, sigma_z]).view(1, -1, 3))

    return (-dist_sq).exp()

def linear_kernel(p_i, p_j, d_feat):
    """Normalized feature inner product. d_feat must be passed in: p_i is a LazyTensor
    of shape (N, 1, d_feat), so p_i.shape[1] is 1, not the feature dimension -- reading
    it off the LazyTensor silently divided by 1 and left the loss d_feat times too big."""
    return (p_i * p_j).sum(-1) / d_feat


def varifold_sp(S1, S2, sigma=1.0):
    x1, p1 = S1
    x2, p2 = S2

    N, D_pos = x1.shape
    M, _ = x2.shape
    _, D_feat = p1.shape

    x_i = LazyTensor(x1.contiguous().view(N, 1, D_pos))
    x_j = LazyTensor(x2.contiguous().view(1, M, D_pos))
    p_i = LazyTensor(p1.contiguous().view(N, 1, D_feat))
    p_j = LazyTensor(p2.contiguous().view(1, M, D_feat))

    K_pos = gaussian_kernel(x_i, x_j, sigma)
    K_feat = linear_kernel(p_i, p_j, D_feat)
    # Normalize by the product of the number of points to prevent extremely large Loss values
    return (K_pos * K_feat).sum(1).sum() / (N * M)

def varifold_sp_anisotropic(S1, S2, sigma_xy=0.02, sigma_z=0.1):
    x1, p1 = S1
    x2, p2 = S2

    N, D_pos  = x1.shape
    M, _      = x2.shape
    _, D_feat = p1.shape

    x_i = LazyTensor(x1.contiguous().view(N, 1, D_pos))
    x_j = LazyTensor(x2.contiguous().view(1, M, D_pos))
    p_i = LazyTensor(p1.contiguous().view(N, 1, D_feat))
    p_j = LazyTensor(p2.contiguous().view(1, M, D_feat))

    K_pos  = gaussian_kernel_anisotropic(x_i, x_j, sigma_xy, sigma_z)
    #K_pos_is = gaussian_kernel(x_i, x_j, sigma_xy)
    #assert sigma_xy == sigma_z
    #assert torch.allclose(K_pos.sum(0), K_pos_is.sum(0))

    K_feat = linear_kernel(p_i, p_j, D_feat)
    # Normalize by the product of the number of points to prevent extremely large Loss values
    return (K_pos * K_feat).sum(1).sum() / (N * M)


def isotropic_sigma(base_sigma, M, dim=3):
    """Bandwidth scaled to the mean spacing of M points filling a unit `dim`-cube:
    M^(-1/dim). The exponent must follow the ambient dimension -- using 1/3 on the 2D
    BARSeq clouds makes sigma ~M^(1/6) too wide (10x at M=1e6), which oversmooths the
    reconstruction and reports an over-optimistic eps."""
    return base_sigma * (1 / M) ** (1 / dim)


def make_varifold_fn(kernel_type, M, sigma, sigma_xy, sigma_z, dim=3):
    """
    Build a varifold callable(S1, S2) -> scalar from the config kernel choice.
    The caller passes the already-computed isotropic bandwidth `sigma`, so both the
    tiled pipeline (sigma = BASE_SIGMA) and the plain pipeline (sigma scaled by M)
    reuse this factory without changing their behaviour.

    `dim` is the ambient dimension of the point positions (X.shape[1]).
    """
    if kernel_type == "isotropic":
        sigma = isotropic_sigma(sigma, M, dim)
        return lambda S1, S2: varifold_sp(S1, S2, sigma)
    if kernel_type == "anisotropic":
        if dim != 3:
            raise ValueError(
                f"KERNEL_TYPE='anisotropic' needs 3D positions (it splits x/y from z), "
                f"but the data is {dim}D. Use 'isotropic'.")
        return lambda S1, S2: varifold_sp_anisotropic(S1, S2, sigma_xy, sigma_z)
    raise ValueError(f"Unknown KERNEL_TYPE: '{kernel_type}'.")


if __name__ == "__main__":
    import torch
    x = torch.randn(5, 1, 3)
    y = torch.randn(1, 5, 3)

    sig = torch.tensor(1.5)

    isotrope = gaussian_kernel(x, y, sig)
    anisotrope = gaussian_kernel_anisotropic(x, y, sig, sig)

    assert torch.allclose(isotrope, anisotrope)


