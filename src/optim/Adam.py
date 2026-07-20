# src/optim/Adam.py
import math
import time

import torch
import torch.nn.functional as F


def _inverse_softplus(P, floor=1e-8):
    """theta with softplus(theta) == P. P is floored so exact zeros map to a large
    finite negative theta instead of -inf."""
    return torch.log(torch.expm1(P.clamp(min=floor)))


def _adam_loop(S_targets, X_hat, P_hat, varifold_fn, lr_X, lr_P, epochs,
               target_eps, min_epochs, stall_window, stall_tol,
               softplus_P, lr_decay, tag):
    """
    Shared Adam driver. Optimizes the measure (X_hat, P_hat) against one or more
    target measures. The single-target case (optimize_adam) is a one-element list;
    the multi-target case (optimize_adam_joint) sums the per-target losses.

    Stopping is governed by three independent criteria, reported on exit:
      * converged — the relative residual eps = ||S - S_hat|| / ||S|| reached target_eps
      * stalled   — the windowed loss average stopped improving (saves the plateau)
      * max epochs — fallback, warns because the target was not reached
    """
    targets = [S for S in S_targets if S[0].shape[0] > 0]
    if not targets:
        return X_hat, P_hat, [], []

    # P is optimized in softplus space: non-negativity is built into the
    # parametrization. The hard clamp_ used previously zeroes the gradient of every
    # point it touches and pins it at 0 forever ("dead" representative points).
    train_P = P_hat.requires_grad          # False when the caller relocates X only
    if softplus_P and train_P:
        theta = _inverse_softplus(P_hat.detach()).requires_grad_(True)
        P_param = theta
        current_P = lambda: F.softplus(theta)
    else:
        P_param = P_hat
        current_P = lambda: P_hat

    optimiser = torch.optim.Adam(
        [{'params': [X_hat], 'lr': lr_X},
         {'params': [P_param], 'lr': lr_P}],
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=stall_window) if lr_decay else None

    self_terms = [varifold_fn(S, S) for S in targets]
    # ||S||^2 — the constant part of the loss. Dividing by it turns the loss into the
    # squared relative residual ||S - S_hat||^2 / ||S||^2: dimensionless, and therefore
    # comparable across tiles whose point counts differ by orders of magnitude.
    norm_sq = float(sum(t.item() for t in self_terms))
    target_loss = (target_eps ** 2) * norm_sq if target_eps > 0 else -math.inf

    loss_history = []
    time_history = []
    best_loss, best_X, best_P = math.inf, None, None
    status = "max epochs"
    print(f"Starting {tag} optimization loop... "
          f"(target eps={target_eps:.4f}, max {epochs} epochs)")
    start_time = time.time()

    for epoch in range(epochs):
        optimiser.zero_grad()
        P_cur = current_P()
        S_hat = (X_hat, P_cur)
        term_hat = varifold_fn(S_hat, S_hat)
        loss = self_terms[0] + term_hat - 2 * varifold_fn(targets[0], S_hat)
        for term0, S in zip(self_terms[1:], targets[1:]):
            loss = loss + term0 + term_hat - 2 * varifold_fn(S, S_hat)

        loss.backward()
        optimiser.step()

        with torch.no_grad():
            X_hat.clamp_(min=0.0, max=1.0)
            if not softplus_P and train_P:
                P_hat.clamp_(min=0)

        loss_val = loss.item()
        loss_history.append(loss_val)
        time_history.append(time.time() - start_time)
        if scheduler is not None:
            scheduler.step(loss_val)

        # Adam drifts after reaching its best point — keep a snapshot and return that.
        if loss_val < best_loss:
            best_loss = loss_val
            with torch.no_grad():
                best_X = X_hat.detach().clone()
                best_P = current_P().detach().clone()

        eps = math.sqrt(max(loss_val, 0.0) / norm_sq) if norm_sq > 0 else float("nan")
        if (epoch + 1) % 10 == 0:
            print(f"[{tag}] Epoch {epoch+1:4d}/{epochs} | "
                  f"Loss: {loss_val:.8f} | rel eps: {eps:.4%}")

        if loss_val <= target_loss:
            status = "converged"
            print(f"[{tag}] Target reached at epoch {epoch+1} | rel eps: {eps:.4%}")
            break

        # Windowed stall test: compare the mean of the last `stall_window` epochs with
        # the mean of the window before it. A single-step delta (the old criterion) is
        # far too jumpy for Adam — its warm-up plateau trips it almost immediately.
        if epoch + 1 >= max(min_epochs, 2 * stall_window):
            prev = sum(loss_history[-2*stall_window:-stall_window]) / stall_window
            cur = sum(loss_history[-stall_window:]) / stall_window
            if (prev - cur) / max(abs(prev), 1e-30) < stall_tol:
                status = "stalled"
                print(f"[{tag}] Stalled at epoch {epoch+1} | rel eps: {eps:.4%} "
                      f"(target {target_eps:.4%} not reached)")
                break

    # Restore the best iterate in place so the caller keeps its own tensor objects.
    if best_X is not None:
        with torch.no_grad():
            X_hat.copy_(best_X)
            if train_P:
                P_hat.copy_(best_P)

    best_eps = math.sqrt(max(best_loss, 0.0) / norm_sq) if norm_sq > 0 else float("nan")
    if status == "max epochs":
        print(f"[{tag}] WARNING: hit the {epochs}-epoch limit without reaching "
              f"target eps={target_eps:.4%} (best {best_eps:.4%})")
    print(f"[{tag}] Done ({status}) | best loss: {best_loss:.8f} | "
          f"best rel eps: {best_eps:.4%} | {time.time() - start_time:.1f} s")

    return X_hat, P_hat, loss_history, time_history


def optimize_adam(S, X_hat, P_hat, varifold_fn, lr_X=0.003, lr_P=0.002, epochs=5000,
                  target_eps=0.05, min_epochs=300, stall_window=50, stall_tol=1e-4,
                  softplus_P=True, lr_decay=True):
    """
    Optimize representative points against a single target S using Adam.
    """
    return _adam_loop([S], X_hat, P_hat, varifold_fn, lr_X, lr_P, epochs,
                      target_eps, min_epochs, stall_window, stall_tol,
                      softplus_P, lr_decay, tag="Adam")


def optimize_adam_joint(S_targets, X_hat, P_hat, varifold_fn, lr_X=0.003, lr_P=0.002,
                        epochs=5000, target_eps=0.05, min_epochs=300, stall_window=50,
                        stall_tol=1e-4, softplus_P=True, lr_decay=True):
    """Optimize one measure jointly against several target measures (shared seam/overlap)."""
    return _adam_loop(S_targets, X_hat, P_hat, varifold_fn, lr_X, lr_P, epochs,
                      target_eps, min_epochs, stall_window, stall_tol,
                      softplus_P, lr_decay, tag="Joint Adam")
