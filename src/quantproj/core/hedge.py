"""Box-constrained hedge fitting with a unique least-movement tie-break."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize


def positions(loadings, target, previous, caps, weights=None):
    """Lexicographic solve in factor or observable quote coordinates.

    The optimal fitted image is unique even if the trading positions are not.
    Minimize strictly positive weighted movement within that image's preimage.
    """
    a = np.asarray(loadings, dtype=float)
    b = np.asarray(target, dtype=float)
    previous = np.asarray(previous, dtype=float)
    caps = np.broadcast_to(np.asarray(caps, dtype=float), previous.shape)
    w = np.ones_like(previous) if weights is None else np.broadcast_to(weights, previous.shape)
    if np.any(w <= 0) or np.any(caps <= 0):
        raise ValueError("Hedge weights and caps must be positive.")
    scale = max(np.linalg.norm(a, ord=np.inf), 1e-14)
    a, b = a / scale, b / scale
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = np.count_nonzero(s > (s[0] * 1e-10 if len(s) else 0))
    if not rank:
        return np.clip(previous, -caps, caps)
    basis = vt[:rank]
    coordinates = (u[:, :rank].T @ b) / s[:rank]
    inv_w = 1.0 / w

    def closest(coords):
        normal = (basis * inv_w) @ basis.T
        return previous + inv_w * (basis.T @ np.linalg.solve(normal, coords - basis @ previous))

    h = closest(coordinates)
    if np.all(abs(h) <= caps + 1e-10):
        return np.clip(h, -caps, caps)
    first = lsq_linear(a, b, bounds=(-caps, caps), tol=1e-12, lsmr_tol=1e-13, max_iter=300)
    if not first.success:
        raise RuntimeError("The constrained hedge residual did not converge.")
    coords = basis @ first.x
    h = closest(coords)
    if np.all(abs(h) <= caps + 1e-9):
        return np.clip(h, -caps, caps)
    if rank == len(previous):
        return first.x

    def loss(x):
        diff = x - previous
        return 0.5 * np.dot(w * diff, diff)

    solve = minimize(
        loss,
        first.x,
        jac=lambda x: w * (x - previous),
        method="SLSQP",
        bounds=Bounds(-caps, caps),
        constraints=LinearConstraint(basis, coords, coords),
        options={"ftol": 1e-11, "maxiter": 250},
    )
    if not solve.success and (
        np.max(abs(basis @ solve.x - coords)) > 1e-8 or np.any(abs(solve.x) > caps + 1e-8)
    ):
        raise RuntimeError("The constrained hedge tie-break did not converge.")
    return np.clip(solve.x, -caps, caps)
