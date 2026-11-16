"""Deployment-only local prefix refitting; no model or hardware dependencies."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline


@dataclass
class PrefixFit:
    future: np.ndarray
    original_control_points: np.ndarray
    control_points: np.ndarray
    history_rmse_before: float
    history_rmse_after: float


def fit_local_prefix(future, history, *, degree=3, control_points=8,
                     free_control_points=3, history_weight=4.0, regularization=0.05):
    """Fit future samples, then change only the first free control points.

    Future indices are 0..N-1, previous target indices -H..-1. The first
    polynomial span is extended backwards ONLY to construct the history fit;
    execution always samples inside 0..N-1. A ridge on coefficient *changes*
    keeps underdetermined prefix fits close to the predicted trajectory.
    This is a soft history fit, not a hard C1/C2 continuity constraint.
    """
    future, history = np.asarray(future, float), np.asarray(history, float)
    if (future.ndim != 2 or history.ndim != 2 or history.shape[1] != future.shape[1]
            or len(history) < 2 or not np.isfinite(future).all() or not np.isfinite(history).all()):
        raise ValueError("future/history must be finite compatible matrices with at least 2 history samples")
    if degree != 3 or not degree + 1 <= control_points <= len(future):
        raise ValueError("local prefix fit needs cubic degree and 4 <= control_points <= horizon")
    if not 1 <= free_control_points < control_points:
        raise ValueError("free_control_points must be in [1, control_points)")
    if (not np.isfinite(history_weight) or history_weight <= 0
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("history_weight must be positive and regularization nonnegative (both finite)")
    x = np.arange(len(future), dtype=float)
    internal = np.linspace(x[0], x[-1], control_points - degree + 1)
    knots = np.r_[[x[0]] * degree, internal, [x[-1]] * degree]
    fitted = make_lsq_spline(x, future, knots, k=degree, axis=0)
    original = fitted.c.copy()
    past_x = np.arange(-len(history), 0, dtype=float)
    # Basis evaluation uses the identical boundary polynomial for all columns.
    basis = BSpline(knots, np.eye(control_points), degree, extrapolate=True)(past_x)
    residual = history - basis @ original
    lhs = history_weight * basis[:, :free_control_points]
    rhs = history_weight * residual
    if regularization:
        lhs = np.vstack([lhs, np.sqrt(regularization) * np.eye(free_control_points)])
        rhs = np.vstack([rhs, np.zeros((free_control_points, future.shape[1]))])
    delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    updated = original.copy()
    updated[:free_control_points] += delta
    rebuilt = BSpline(knots, updated, degree, extrapolate=False)(x)
    if not np.isfinite(rebuilt).all():
        raise ValueError("local prefix refit produced nonfinite actions")
    return PrefixFit(rebuilt, original, updated,
                     float(np.sqrt(np.mean(residual ** 2))),
                     float(np.sqrt(np.mean((history - basis @ updated) ** 2))))
