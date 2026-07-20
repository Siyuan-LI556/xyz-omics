# src/optim/LBFGS.py
import torch
import time


def _lbfgs_loop(S_targets, X_hat, P_hat, varifold_fn, lr, max_iter, history_size,
                epochs, tol, patience, tag):
    """
    Shared LBFGS driver. Optimizes the measure (X_hat, P_hat) against one or more
    target measures. The single-target case (optimize_lbfgs) is just a one-element
    list; the multi-target case (optimize_lbfgs_joint) sums the per-target losses.
    """
    targets = [S for S in S_targets if S[0].shape[0] > 0]
    if not targets:
        return X_hat, P_hat, [], []

    optimiser = torch.optim.LBFGS(
        [X_hat, P_hat],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe"
    )

    self_terms = [varifold_fn(S, S) for S in targets]

    def closure():
        optimiser.zero_grad()
        S_hat = (X_hat, P_hat)
        term_hat = varifold_fn(S_hat, S_hat)
        loss = self_terms[0] + term_hat - 2 * varifold_fn(targets[0], S_hat)
        for term0, S in zip(self_terms[1:], targets[1:]):
            loss = loss + term0 + term_hat - 2 * varifold_fn(S, S_hat)
        loss.backward()
        return loss

    loss_history = []
    time_history = []
    no_improve_count = 0
    print(f"Starting {tag} optimization loop...")
    start_time = time.time()

    for epoch in range(epochs):
        loss = optimiser.step(closure)
        with torch.no_grad():
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)
        loss_history.append(loss.item())
        time_history.append(time.time() - start_time)
        print(f"[{tag}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.8f}")

        if len(loss_history) > 1:
            delta = abs(loss_history[-2] - loss_history[-1])
            if delta < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1:3d} | Loss: {loss_history[-1]:.8f} < tol {tol:.2e}")
                break

    return X_hat, P_hat, loss_history, time_history


def optimize_lbfgs(S, X_hat, P_hat, varifold_fn, lr, max_iter, history_size, epochs, tol=1e-6, patience=5):
    """
    Optimize representative points against a single target S using LBFGS
    with strong Wolfe line search.

    Parameters
    ----------
    S            : (X, P) tuple of the original point cloud
    X_hat, P_hat : initial representative positions and features (requires_grad)
    varifold_fn  : callable(S1, S2) -> scalar  — isotropic or anisotropic varifold
    """
    return _lbfgs_loop([S], X_hat, P_hat, varifold_fn, lr, max_iter, history_size,
                       epochs, tol, patience, tag="LBFGS")


def optimize_lbfgs_joint(S_targets, X_hat, P_hat, varifold_fn, lr, max_iter, history_size, epochs, tol=1e-6, patience=5):
    """Optimize one measure jointly against several target measures (shared seam/overlap)."""
    return _lbfgs_loop(S_targets, X_hat, P_hat, varifold_fn, lr, max_iter, history_size,
                       epochs, tol, patience, tag="Joint LBFGS")
