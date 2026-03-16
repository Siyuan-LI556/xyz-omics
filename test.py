import numpy as np
import torch
import torch.optim as optim
import pyvista as pv


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


print("Loading .npz file...")
#data = np.load("AllenAtlas_approx200um_flipZ.npz")
data = np.load("D076_1L_approx200um.npz")

# Extract spatial coordinates (62453, 3) and gene features (62453, 39)
#X_np = data['Z']
#P_np = data['nu_Z']

X_np = data['X']
P_np = data['nu_X']

# Convert to PyTorch tensors
X_orig = torch.tensor(X_np, dtype=torch.float32)
P_orig = torch.tensor(P_np, dtype=torch.float32)

X_min, _ = torch.min(X_orig, dim=0)
X_max, _ = torch.max(X_orig, dim=0)
X_orig = (X_orig - X_min) / (X_max - X_min)
S = (X_orig, P_orig)


N = X_orig.shape[0]  
M = 2000             

print(f"Preparing to compress {N} brain cell points into {M} representative points...")

indices = torch.randperm(N)[:M]
#indices = torch.linspace(0, N - 1, M).long()
X_hat = X_orig[indices].clone()
P_hat = P_orig[indices].clone()

X_hat.requires_grad_(True)
P_hat.requires_grad_(True)
S_hat = (X_hat, P_hat)

optimizer = optim.Adam([
    {'params': [X_hat], 'lr': 0.01},  
    {'params': [P_hat], 'lr': 0.005}    
    ])

epochs = 50
sigma = 0.10

print("Starting optimization loop...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    term1 = varifold_sp(S_hat, S_hat, sigma)
    term2 = varifold_sp(S, S_hat, sigma)
    loss = term1 - 2 * term2 
    
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

cloud_orig.point_data["Gene_0"] = P_np[:, 0]
cloud_hat.point_data["Gene_0"] = p_hat_final[:, 0]

cloud_orig.save("1.4-62k.vtp")
cloud_hat.save("1.4-1k.vtp")
