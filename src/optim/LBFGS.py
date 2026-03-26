# src/optim/LBFGS.py
import torch
from src.losses.varifold import varifold_sp


def optimize_lbfgs(S, X_hat, P_hat, bandwidth_varifold, lr, max_iter, history_size, epochs):
    """
    Optimize representative points using LBFGS with strong Wolfe line search.
    """
    optimiser = torch.optim.LBFGS(
        [X_hat, P_hat],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe"
    )

    term0 = varifold_sp(S, S, bandwidth_varifold)

    def closure():
        optimiser.zero_grad()
        S_hat = (X_hat, P_hat)
        term1 = varifold_sp(S_hat, S_hat, bandwidth_varifold)
        term2 = varifold_sp(S, S_hat, bandwidth_varifold)
        loss = term0 + term1 - 2 * term2
        loss.backward()
        return loss

    loss_history = []
    print("Starting LBFGS optimization loop...")
    for epoch in range(epochs):
        loss = optimiser.step(closure)
        with torch.no_grad():
            # Constrain gene expression to a reasonable range (assuming expression cannot be negative)
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)
        loss_history.append(loss.item())
        print(f"[LBFGS] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

    return X_hat, P_hat, loss_history