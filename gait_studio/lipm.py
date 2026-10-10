"""Linear inverted pendulum CoM planning by ZMP preview control (Kajita et al., ICRA 2003).

The cart-table model with jerk input is discretised at ``dt``; an integral-action LQ preview
controller (Katayama 1985 form) tracks a ZMP reference ``horizon`` seconds ahead. x and y are
independent, so both axes share one set of gains. r is lower than Kajita's 1e-6 because short
double-support windows need the ZMP to transfer quickly; CoM acceleration barely changes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_discrete_are

G = 9.81


@lru_cache(maxsize=32)
def _gains(zc: float, dt: float, horizon: float, q_e: float, r: float):
    A = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    B = np.array([[dt ** 3 / 6.0], [dt * dt / 2.0], [dt]])
    C = np.array([[1.0, 0.0, -zc / G]])

    # Augmented system: state [sum of ZMP error ; x]
    At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    Bt = np.vstack([C @ B, B])
    It = np.vstack([np.eye(1), np.zeros((3, 1))])
    Q = np.zeros((4, 4))
    Q[0, 0] = q_e
    R = np.array([[r]])

    P = solve_discrete_are(At, Bt, Q, R)
    S = np.linalg.inv(R + Bt.T @ P @ Bt)
    K = S @ Bt.T @ P @ At
    Ki, Kx = float(K[0, 0]), K[0, 1:]

    n_prev = int(round(horizon / dt))
    Ac = At - Bt @ K
    X = -Ac.T @ P @ It
    Gd = np.zeros(n_prev)
    Gd[0] = -Ki
    for j in range(1, n_prev):
        Gd[j] = float(S @ Bt.T @ X)
        X = Ac.T @ X
    return A, B, C, Ki, Kx, Gd


def preview_com(zmp_ref: np.ndarray, zc: float, dt: float, x0: np.ndarray | None = None,
                horizon: float = 1.2, q_e: float = 1.0, r: float = 1e-8) -> np.ndarray:
    """Plan a CoM trajectory tracking ``zmp_ref``.

    zmp_ref: (T, 2) reference sampled at dt; must extend ``horizon`` past the span you use.
    x0:      (2, 3) initial [pos, vel, acc] per axis.
    Returns  (T, 2, 3) CoM [pos, vel, acc] per axis; the last ``horizon`` seconds hold the
             reference constant and should be discarded by the caller.
    """
    A, B, C, Ki, Kx, Gd = _gains(float(zc), float(dt), float(horizon), float(q_e), float(r))
    T = zmp_ref.shape[0]
    n_prev = Gd.shape[0]
    pad = np.vstack([zmp_ref, np.repeat(zmp_ref[-1:], n_prev + 1, axis=0)])

    x = np.zeros((3, 2)) if x0 is None else np.asarray(x0, dtype=float).T.copy()  # (3, axes)
    err_sum = np.zeros(2)
    out = np.zeros((T, 2, 3))
    for k in range(T):
        out[k] = x.T
        p = (C @ x).ravel()
        err_sum += p - pad[k]
        u = -Ki * err_sum - Kx @ x - Gd @ pad[k + 1: k + 1 + n_prev]
        x = A @ x + B @ u[None, :]
    return out
