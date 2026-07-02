# src/optim/Adam.py
import torch
import time


def optimize_adam(S, X_hat, P_hat, varifold_fn, lr_X=0.01, lr_P=0.005, epochs=5000, tol=1e-6, patience=50):
    """
    Optimize representative points using Adam.

    Parameters
    ----------
    S            : (X, P) tuple of the original point cloud
    X_hat, P_hat : initial representative positions and features (requires_grad)
    varifold_fn  : callable(S1, S2) -> scalar  — isotropic or anisotropic varifold
    lr_X         : learning rate for X_hat
    lr_P         : learning rate for P_hat
    epochs       : maximum number of gradient steps
    tol          : early-stopping tolerance on loss change
    patience     : consecutive epochs below tol before stopping
    """
    optimiser = torch.optim.Adam([
        {'params': [X_hat], 'lr': lr_X},
        {'params': [P_hat], 'lr': lr_P},
    ])

    term0 = varifold_fn(S, S)

    loss_history = []
    time_history = []
    no_improve_count = 0
    print("Starting Adam optimization loop...")
    start_time = time.time()

    for epoch in range(epochs):
        optimiser.zero_grad()

        S_hat = (X_hat, P_hat)
        term1 = varifold_fn(S_hat, S_hat)
        term2 = varifold_fn(S, S_hat)
        loss = term0 + term1 - 2 * term2

        loss.backward()
        optimiser.step()

        with torch.no_grad():
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)

        loss_val = loss.item()
        loss_history.append(loss_val)
        time_history.append(time.time() - start_time)

        if (epoch + 1) % 10 == 0:
            print(f"[Adam] Epoch {epoch+1:4d}/{epochs} | Loss: {loss_val:.8f}")

        if len(loss_history) > 1:
            delta = abs(loss_history[-2] - loss_history[-1])
            if delta < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1:4d} | Loss: {loss_val:.8f} < tol {tol:.2e}")
                break

    return X_hat, P_hat, loss_history, time_history


def optimize_adam_joint(S_targets, X_hat, P_hat, varifold_fn, lr_X=0.01, lr_P=0.005, epochs=5000, tol=1e-6, patience=50):
    targets = [S for S in S_targets if S[0].shape[0] > 0]
    if not targets:
        return X_hat, P_hat, [], []

    optimiser = torch.optim.Adam([
        {'params': [X_hat], 'lr': lr_X},
        {'params': [P_hat], 'lr': lr_P},
    ])

    target_self_terms = [varifold_fn(S, S) for S in targets]
    loss_history = []
    time_history = []
    no_improve_count = 0
    print("Starting joint Adam optimization loop...")
    start_time = time.time()

    for epoch in range(epochs):
        optimiser.zero_grad()
        S_hat = (X_hat, P_hat)
        term_hat = varifold_fn(S_hat, S_hat)
        loss = target_self_terms[0] + term_hat - 2 * varifold_fn(targets[0], S_hat)
        for term0, S in zip(target_self_terms[1:], targets[1:]):
            loss = loss + term0 + term_hat - 2 * varifold_fn(S, S_hat)

        loss.backward()
        optimiser.step()

        with torch.no_grad():
            P_hat.clamp_(min=0)
            X_hat.clamp_(min=0.0, max=1.0)

        loss_val = loss.item()
        loss_history.append(loss_val)
        time_history.append(time.time() - start_time)

        if (epoch + 1) % 10 == 0:
            print(f"[Joint Adam] Epoch {epoch+1:4d}/{epochs} | Loss: {loss_val:.8f}")

        if len(loss_history) > 1:
            delta = abs(loss_history[-2] - loss_history[-1])
            if delta < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1:4d} | Loss: {loss_val:.8f} < tol {tol:.2e}")
                break

    return X_hat, P_hat, loss_history, time_history
