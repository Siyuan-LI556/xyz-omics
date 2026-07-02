import torch
import os
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "BARSeq", "D076_1L_approx200um.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)
X_orig,X_hat,P_hat,P_orig, loss_history=torch.load(os.path.join(OUTPUT_DIR, "results.pt",))
print(P_hat.shape)

X_orig_np = X_orig.cpu().numpy()
X_hat_np  = X_hat.detach().cpu().numpy()

# plot p-hat sum(0）
plt.figure()
plt.plot((P_hat.sum(axis=0) /P_hat.sum()).cpu().detach().numpy())
plt.plot((P_orig.sum(axis=0)/P_orig.sum()).cpu().detach().numpy(),"--")
plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "P_hat.png"))
plt.figure()
plt.plot((P_hat.sum(axis=0) /P_hat.sum()-P_orig.sum(axis = 0)/P_orig.sum()).cpu().detach().numpy())
plt.show()