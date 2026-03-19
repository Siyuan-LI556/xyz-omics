# %%
import os.path
import numpy as np
import torch
#import torch.optim as optim
import pyvista as pv
from pykeops.torch import LazyTensor
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Set Paths
data_dir = os.path.join("..", "data", "BARSeq")
output_dir = os.path.join(data_dir, "output")
os.makedirs(output_dir, exist_ok=True)


# %% Declare kernel functions and varifold loss

def gaussian_kernel(x_i, x_j, sigma):
    dist_sq = ((x_i - x_j) ** 2).sum(-1)
    return (-dist_sq / (2 * sigma ** 2)).exp()


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


# %% Load data
print("Loading .npz file...")
data_dir = os.path.join("..", "data", "BARSeq")

data = np.load(os.path.join(data_dir, "D076_1L_approx200um.npz"))

# Barseq data contains 62,453 points with 39 gene features each
Position_BarSeq = data['X']  # a.ka. x_i in xIV-LDDMM paper
Feature_BarSeq = data['nu_X']  # a.ka. w_i*p_i in xIV-LDDMM paper, total gene expression per cell

print(f"Loaded data with {Position_BarSeq.shape[0]} points and {Feature_BarSeq.shape[1]} gene features.")
print(f"Feature range is {Feature_BarSeq.min()}, {Feature_BarSeq.max()}")

# Allen atlas.
# data = np.load("AllenAtlas_approx200um_flipZ.npz")
# Z_np = data['Z']
# P_np = data['nu_Z']

# Convert to PyTorch tensors
X_orig = torch.tensor(Position_BarSeq, dtype=torch.float32, device=device)
P_orig = torch.tensor(Feature_BarSeq, dtype=torch.float32, device=device)

# %% Normalize spatial coordinates to [0, 1] range
X_min, _ = torch.min(X_orig, dim=0)
X_max, _ = torch.max(X_orig, dim=0)
X_orig = (X_orig - X_min) / (X_max - X_min)
S = (X_orig, P_orig)

# bandwidth_varifold = 0.10 # check dependency to M
base_sigma = 1.0  # Base bandwidth for varifold kernel, can be tuned based on data scale and desired smoothness.

# %% Subsample data
N = X_orig.shape[0]
M = 1000
bandwidth_varifold = base_sigma * (1 / M) ** (1 / 3)
print(f"bandwidth_varifold set to {bandwidth_varifold:.4f} based on M={M} representative points.")
print(f"Preparing to compress {N} brain cell points into {M} representative points...")

''' Subsample random
indices = torch.randperm(N)[:M]

X_hat = X_orig[indices].clone()
P_hat = P_orig[indices].clone()

X_hat.requires_grad_(True)
P_hat.requires_grad_(True)
S_hat = (X_hat, P_hat)
'''
#  Implement more sophisticated subsampling strategies to better capture the spatial distribution and gene expression
#  diversity of the original dataset. Random sampling may not be sufficient to preserve important biological patterns in the data.

# Subsample by K-means
kmeans = KMeans(n_clusters=M, random_state=0)
labels = kmeans.fit_predict(X_orig.cpu().numpy())
centers = torch.tensor(kmeans.cluster_centers_, device=device)

P_hat = torch.zeros((M, P_orig.shape[1]), device=device)

for i in range(M):
    mask = torch.tensor(labels == i, device=device)
    if mask.sum() > 0:
        P_hat[i] = P_orig[mask].mean(dim=0)

X_hat = centers.clone().requires_grad_(True)
P_hat = P_hat.clone().requires_grad_(True)
S_hat = (X_hat, P_hat)

# %% Set optimization parameters

'''optimizer Adam
optimizer = optim.Adam([
    {'params': [X_hat], 'lr': 0.01},
    {'params': [P_hat], 'lr': 0.005}
])

# TODO: Plot loss decreasing. Tune optimizer hyperparameters (learning rates, weight decay, etc.) to improve convergence and final results.
epochs = 5000

# %% Optimization loop
print("Starting optimization loop...")

term0 = varifold_sp(S, S, bandwidth_varifold)


for epoch in range(epochs):
    optimizer.zero_grad()

    term1 = varifold_sp(S_hat, S_hat, bandwidth_varifold)
    term2 = varifold_sp(S, S_hat, bandwidth_varifold)
    loss = term0 + term1 - 2 * term2

    loss.backward()
    optimizer.step()

    # Constrain gene expression to a reasonable range (assuming expression cannot be negative)
    with torch.no_grad():
        P_hat.clamp_(min=0)
        X_hat.clamp_(min=0.0, max=1.0)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}/{epochs} | Varifold Loss: {loss.item():.6f}")
'''
# optimizers LBFGS
optimiser = torch.optim.LBFGS(
    [X_hat, P_hat],
    lr=0.1,
    max_iter=20,
    history_size=10,
    line_search_fn="strong_wolfe"
)

term0 = varifold_sp(S, S, bandwidth_varifold)

def closure():
    optimiser.zero_grad()

    S_hat = (X_hat, P_hat)

    term1 = varifold_sp(S_hat, S_hat, bandwidth_varifold)
    term2 = varifold_sp(S, S_hat, bandwidth_varifold)
    loss = term0 + term1 - 2*term2

    loss.backward()
    return loss

epochs = 50
loss_history = []
print("Starting LBFGS optimization loop...")
for epoch in range(epochs):
    loss = optimiser.step(closure)

    with torch.no_grad():
        P_hat.clamp_(min=0)
        X_hat.clamp_(min=0.0, max=1.0)

    loss_history.append(loss.item())

    print(f"[LBFGS] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

print("\nOptimization complete! Exporting ParaView format files...")

X_hat_restored = X_hat.detach() * (X_max - X_min) + X_min
X_orig_restored = X_orig.detach() * (X_max - X_min) + X_min

x_orig_final = X_orig_restored.cpu().numpy()
x_hat_final = X_hat_restored.cpu().numpy()
p_hat_final = P_hat.detach().cpu().numpy()

cloud_orig = pv.PolyData(x_orig_final)
cloud_hat = pv.PolyData(x_hat_final)

# TODO: Fix export : export all feature and/or new summary feature for visualization to check quality: e.g.
cloud_orig.point_data["Gene_weight"] = Feature_BarSeq#.sum(axis=1)
cloud_hat.point_data["Gene_weight"] = p_hat_final#.sum(axis=1)

file_orig = os.path.join(output_dir, "orig_1.4.vtp")
file_hat = os.path.join(output_dir, "hat_1.4.vtp")

cloud_orig.save(file_orig)
cloud_hat.save(file_hat)