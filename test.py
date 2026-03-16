#%%
import os.path

import numpy as np
import torch
import torch.optim as optim
import pyvista as pv


device = torch.device("cuda" if torch.cuda.is_available() * 0. else "cpu")
print(f"Using device: {device}")


#%% Set Paths
data_dir = os.path.join("..", "data", "BARSeq")
output_dir = os.path.join(data_dir, "output")
os.makedirs(output_dir, exist_ok=True)



#%% Declare kernel functions and varifold loss

def gaussian_kernel(x, y, sigma):
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist_sq / (2 * sigma**2))

def linear_kernel(p, q):
    return torch.matmul(p, q.T)/ p.shape[1]

def varifold_sp(S1, S2, sigma=1.0):
    x1, p1 = S1
    x2, p2 = S2
    K_pos = gaussian_kernel(x1, x2, sigma)
    K_feat = linear_kernel(p1, p2)
    # Normalize by the product of the number of points to prevent extremely large Loss values
    return torch.sum(K_pos * K_feat) / (x1.shape[0] * x2.shape[0])

#%% Load data
print("Loading .npz file...")
data_dir = os.path.join("..", "data", "BARSeq")

data = np.load(os.path.join(data_dir, "D076_1L_approx200um.npz"))

# Barseq data contains 62,453 points with 39 gene features each
Position_BarSeq = data['X'] # a.ka. x_i in xIV-LDDMM paper
Feature_BarSeq = data['nu_X']  #  a.ka. w_i*p_i in xIV-LDDMM paper, total gene expression per cell

print(f"Loaded data with {Position_BarSeq.shape[0]} points and {Feature_BarSeq.shape[1]} gene features.")
print(f"Feature range is {Feature_BarSeq.min()}, {Feature_BarSeq.max()}")

# Allen atlas.
#data = np.load("AllenAtlas_approx200um_flipZ.npz")
#Z_np = data['Z']
#P_np = data['nu_Z']

# Convert to PyTorch tensors
X_orig = torch.tensor(Position_BarSeq, dtype=torch.float32, device=device)
P_orig = torch.tensor(Feature_BarSeq, dtype=torch.float32, device=device)

#%% Normalize spatial coordinates to [0, 1] range
X_min, _ = torch.min(X_orig, dim=0)
X_max, _ = torch.max(X_orig, dim=0)
X_orig = (X_orig - X_min) / (X_max - X_min)
S = (X_orig, P_orig)

bandwith_varifold = 0.10 # TODO : check dependency to M


#%% Subsample data
N = X_orig.shape[0]  
M = 2000
print(f"Preparing to compress {N} brain cell points into {M} representative points...")

# TODO: Implement more sophisticated subsampling strategies to better capture the spatial distribution and gene expression diversity of the original dataset. Random sampling may not be sufficient to preserve important biological patterns in the data.
indices = torch.randperm(N)[:M]
#indices = torch.linspace(0, N - 1, M).long()
X_hat = X_orig[indices].clone()
P_hat = P_orig[indices].clone()

X_hat.requires_grad_(True)
P_hat.requires_grad_(True)
S_hat = (X_hat, P_hat)

#%% Set optimization parameters
# TODO: Try other optimizers LBFGS
optimizer = optim.Adam([
    {'params': [X_hat], 'lr': 0.01},  
    {'params': [P_hat], 'lr': 0.005}    
    ])

# TODO: Plot loss decreasing. Tune optimizer hyperparameters (learning rates, weight decay, etc.) to improve convergence and final results.
epochs = 50

#%% Optimization loop
print("Starting optimization loop...")

term0 = varifold_sp(S, S, bandwith_varifold)
for epoch in range(epochs):
    optimizer.zero_grad()
    
    term1 = varifold_sp(S_hat, S_hat, bandwith_varifold)
    term2 = varifold_sp(S, S_hat, bandwith_varifold)
    loss = term0 + term1 - 2 * term2
    
    loss.backward()
    optimizer.step()
    
    # Constrain gene expression to a reasonable range (assuming expression cannot be negative)
    with torch.no_grad():
        P_hat.clamp_(min=0)
        X_hat.clamp_(min=0.0, max=1.0)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Varifold Loss: {loss.item():.6f}")

print("\nOptimization complete! Exporting ParaView format files...")

X_hat_restored = X_hat.detach() * (X_max - X_min) + X_min
X_orig_restored = X_orig.detach() * (X_max - X_min) + X_min

x_orig_final = X_orig_restored.cpu().numpy()
x_hat_final = X_hat_restored.cpu().numpy()
p_hat_final = P_hat.detach().cpu().numpy()

cloud_orig = pv.PolyData(x_orig_final)
cloud_hat = pv.PolyData(x_hat_final)

# TODO: Fix export : export all feature and/or new summary feature for visualization to check quality: e.g.
cloud_orig.point_data["Gene_weight"] = Feature_BarSeq.sum(dim=1)
cloud_hat.point_data["Gene_weight"] = p_hat_final.sum(dim=1)

cloud_orig.save("1.4-62k.vtp")
cloud_hat.save("1.4-1k.vtp")
