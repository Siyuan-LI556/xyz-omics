# src/io/plot.py
import matplotlib.pyplot as plt
import numpy as np

def plot_gene_distributions(P_orig, P_hat, gene_names=None, output_dir=None):
    """
    Compare gene expression distributions before and after compression.
    """
    n_genes = P_orig.shape[1]
    cols = 6
    rows = (n_genes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    axes = axes.flatten()

    P_orig_np = P_orig.detach().cpu().numpy()
    P_hat_np  = P_hat.detach().cpu().numpy()

    for i in range(n_genes):
        ax = axes[i]
        x_max = np.percentile(P_orig_np[:, i], 95)
        data_orig = P_orig_np[:, i][P_orig_np[:, i] <= x_max]
        data_hat = P_hat_np[:, i][P_hat_np[:, i] <= x_max]

        ax.hist(data_orig, bins=50, alpha=0.5, label="Original", density=False)
        ax.hist(data_hat, bins=50, alpha=0.5, label="Compressed", density=False)
        ax.set_xlim(0, x_max)

        title = gene_names[i] if gene_names else f"Gene {i}"
        ax.set_title(title)
        ax.legend(fontsize=6)

    for j in range(n_genes, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Gene Expression Distribution: Original vs Compressed")
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/gene_distributions_random.png", dpi=150)
    plt.show()

def plot_loss_curve(loss_history, time_history, output_dir=None, suffix=""):
    """
    Plot the loss curve of the optimizer.
    x-axis: time, y-axis: varifold loss value
    """
    plt.figure(figsize=(8, 5))
    plt.plot(time_history, loss_history)
    plt.xscale("log")
    plt.yscale("log")
    #plt.xlim(1, 80)
    #plt.ylim(1e-4, 6e-2)
    plt.xlabel("time(seconds)")
    plt.ylabel("varifold loss")
    plt.title("loss curve")
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/loss_curve_{suffix}.png", dpi=150)
    plt.show()
