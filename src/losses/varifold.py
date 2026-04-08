from pykeops.torch import LazyTensor

# %% Declare kernel functions and varifold loss
def gaussian_kernel(x_i, x_j, sigma):
    dist_sq = ((x_i - x_j) ** 2).sum(-1)
    return (-dist_sq / (2 * sigma ** 2)).exp()

# fonction de gaussian anisotropic
def gaussian_kernel_anisotropic(x_i, x_j, sigma_xy, sigma_z):
    """
    Anisotropic Gaussian kernel with different bandwidths for xy-plane and z-axis.
    Diagonal covariance: diag(sigma_xy^2, sigma_xy^2, sigma_z^2)
    """
    diff   = x_i - x_j
    diff_x = diff[..., 0:1]                                  # x component
    diff_y = diff[..., 1:2]                                  # y component
    diff_z = diff[..., 2:3]                                  # z component

    dist_sq = (diff_x ** 2 + diff_y ** 2) / (2 * sigma_xy ** 2) + \
              (diff_z ** 2)               / (2 * sigma_z  ** 2)

    return (-dist_sq).sum(-1).exp()

def linear_kernel(p_i, p_j):
    return (p_i * p_j).sum(-1) / p_i.shape[1]


def varifold_sp(S1, S2, sigma=1.0):
    x1, p1 = S1
    x2, p2 = S2

    N, D_pos = x1.shape
    M, _ = x2.shape
    _, D_feat = p1.shape

    x_i = LazyTensor(x1.view(N, 1, D_pos))
    x_j = LazyTensor(x2.view(1, M, D_pos))
    p_i = LazyTensor(p1.view(N, 1, D_feat))
    p_j = LazyTensor(p2.view(1, M, D_feat))

    K_pos = gaussian_kernel(x_i, x_j, sigma)
    K_feat = linear_kernel(p_i, p_j)
    # Normalize by the product of the number of points to prevent extremely large Loss values
    return (K_pos * K_feat).sum(1).sum() / (N * M)

def varifold_sp_anisotropic(S1, S2, sigma_xy=0.02, sigma_z=0.1):
    x1, p1 = S1
    x2, p2 = S2

    N, D_pos  = x1.shape
    M, _      = x2.shape
    _, D_feat = p1.shape

    x_i = LazyTensor(x1.view(N, 1, D_pos))
    x_j = LazyTensor(x2.view(1, M, D_pos))
    p_i = LazyTensor(p1.view(N, 1, D_feat))
    p_j = LazyTensor(p2.view(1, M, D_feat))

    K_pos  = gaussian_kernel_anisotropic(x_i, x_j, sigma_xy, sigma_z)
    K_feat = linear_kernel(p_i, p_j)
    # Normalize by the product of the number of points to prevent extremely large Loss values
    return (K_pos * K_feat).sum(1).sum() / (N * M)