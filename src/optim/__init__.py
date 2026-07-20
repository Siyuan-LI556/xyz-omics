from src import config
from .LBFGS import optimize_lbfgs, optimize_lbfgs_joint
from .Adam import optimize_adam, optimize_adam_joint


def run_optimizer(S, X_hat, P_hat, varifold_fn):
    """Optimize one measure against target S using the configured optimizer."""
    if config.OPTIMIZER == "lbfgs":
        return optimize_lbfgs(
            S, X_hat, P_hat, varifold_fn,
            lr=config.LR, max_iter=config.MAX_ITER, history_size=config.HISTORY_SIZE,
            epochs=config.EPOCHS, tol=config.TOL, patience=config.PATIENCE,
        )
    if config.OPTIMIZER == "adam":
        return optimize_adam(
            S, X_hat, P_hat, varifold_fn,
            lr_X=config.ADAM_LR_X, lr_P=config.ADAM_LR_P, epochs=config.ADAM_EPOCHS,
            target_eps=config.ADAM_TARGET_EPS, min_epochs=config.ADAM_MIN_EPOCHS,
            stall_window=config.ADAM_STALL_WINDOW, stall_tol=config.ADAM_STALL_TOL,
            softplus_P=config.ADAM_SOFTPLUS_P, lr_decay=config.ADAM_LR_DECAY,
        )
    raise ValueError(f"Unknown OPTIMIZER: '{config.OPTIMIZER}'.")


def run_optimizer_joint(S_targets, X_hat, P_hat, varifold_fn):
    """Optimize one measure jointly against several targets (used for shared overlaps)."""
    if config.OPTIMIZER == "lbfgs":
        return optimize_lbfgs_joint(
            S_targets, X_hat, P_hat, varifold_fn,
            lr=config.LR, max_iter=config.MAX_ITER, history_size=config.HISTORY_SIZE,
            epochs=config.EPOCHS, tol=config.TOL, patience=config.PATIENCE,
        )
    if config.OPTIMIZER == "adam":
        return optimize_adam_joint(
            S_targets, X_hat, P_hat, varifold_fn,
            lr_X=config.ADAM_LR_X, lr_P=config.ADAM_LR_P, epochs=config.ADAM_EPOCHS,
            target_eps=config.ADAM_TARGET_EPS, min_epochs=config.ADAM_MIN_EPOCHS,
            stall_window=config.ADAM_STALL_WINDOW, stall_tol=config.ADAM_STALL_TOL,
            softplus_P=config.ADAM_SOFTPLUS_P, lr_decay=config.ADAM_LR_DECAY,
        )
    raise ValueError(f"Unknown OPTIMIZER: '{config.OPTIMIZER}'.")
