# src/optim/LBFGS.py
import torch
import time


def optimize_lbfgs(S, X_hat, P_hat, varifold_fn, lr, max_iter, history_size, epochs, tol=1e-6, patience=5):
    """
    Optimize representative points using LBFGS with strong Wolfe line search.

    Parameters
    ----------
    S            : (X, P) tuple of the original point cloud
    X_hat, P_hat : initial representative positions and features (requires_grad)
    varifold_fn  : callable(S1, S2) -> scalar  — isotropic or anisotropic varifold
    """
    optimiser = torch.optim.LBFGS(
        [X_hat, P_hat],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe"
    )

    term0 = varifold_fn(S, S)
    #print(f"term0 is {term0}")
    def closure():
        optimiser.zero_grad()
        S_hat = (X_hat, P_hat)
        term1 = varifold_fn(S_hat, S_hat)
        term2 = varifold_fn(S, S_hat)
        loss  = term0 + term1 - 2 * term2
        loss.backward()
        return loss
    '''
    # Compute gradient at the initial point before any optimizer step.
    # Both kernels are evaluated at the same X_hat, so this is a fair comparison.
    optimiser.zero_grad()
    with torch.enable_grad():
        S_hat0 = (X_hat, P_hat)
        t1 = varifold_fn(S_hat0, S_hat0)
        t2 = varifold_fn(S, S_hat0)
        (term0 + t1 - 2 * t2).backward()
    print(f"[Grad init] X_hat.grad norm: {X_hat.grad.norm().item():.6e}")
    print(f"[Grad init] X_hat.grad[:3]:\n{X_hat.grad[:3]}")
    optimiser.zero_grad()
    '''
    loss_history = []
    time_history = []
    no_improve_count = 0
    print("Starting LBFGS optimization loop...")
    start_time = time.time()

    for epoch in range(epochs):
        loss = optimiser.step(closure)
        with torch.no_grad():
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)
        loss_history.append(loss.item())
        time_history.append(time.time() - start_time)
        print(f"[LBFGS] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.8f}")

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


def optimize_lbfgs_joint(S_targets, X_hat, P_hat, varifold_fn, lr, max_iter, history_size, epochs, tol=1e-6, patience=5):
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

    target_self_terms = [varifold_fn(S, S) for S in targets]

    def closure():
        optimiser.zero_grad()
        S_hat = (X_hat, P_hat)
        term_hat = varifold_fn(S_hat, S_hat)
        loss = target_self_terms[0] + term_hat - 2 * varifold_fn(targets[0], S_hat)
        for term0, S in zip(target_self_terms[1:], targets[1:]):
            loss = loss + term0 + term_hat - 2 * varifold_fn(S, S_hat)
        loss.backward()
        return loss

    loss_history = []
    time_history = []
    no_improve_count = 0
    print("Starting joint LBFGS optimization loop...")
    start_time = time.time()

    for epoch in range(epochs):
        loss = optimiser.step(closure)
        with torch.no_grad():
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)
        loss_history.append(loss.item())
        time_history.append(time.time() - start_time)
        print(f"[Joint LBFGS] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.8f}")

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
