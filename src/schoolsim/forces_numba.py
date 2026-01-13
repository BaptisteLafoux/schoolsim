"""Numba-accelerated force calculations."""
import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree


@njit(cache=True)
def _compute_neighbor_forces(
    X: np.ndarray,  # (2, N)
    V: np.ndarray,  # (2, N)
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    a: float,
    Ra: float,
    J: float,
    n_fish: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused attraction + alignment computation in one pass."""
    F_attr = np.zeros((2, n_fish), dtype=np.float32)
    F_align = np.zeros((2, n_fish), dtype=np.float32)
    n_neighbors = np.zeros(n_fish, dtype=np.int32)

    Ra2 = Ra * Ra

    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]

        # Position difference (j -> i direction)
        dx = X[0, j] - X[0, i]
        dy = X[1, j] - X[1, i]

        # Distance
        r = math.sqrt(dx * dx + dy * dy)
        r_safe = max(r, 1e-10)
        inv_r = 1.0 / r_safe
        inv_r3 = inv_r * inv_r * inv_r

        # Attraction: Xij/r - Ra² * Xij/r³
        F_attr[0, i] += dx * inv_r - Ra2 * dx * inv_r3
        F_attr[1, i] += dy * inv_r - Ra2 * dy * inv_r3

        # Alignment: Vj - Vi
        F_align[0, i] += V[0, j] - V[0, i]
        F_align[1, i] += V[1, j] - V[1, i]

        n_neighbors[i] += 1

    return F_attr, F_align, n_neighbors


@njit(parallel=True, cache=True)
def _normalize_forces(
    F_attr: np.ndarray,
    F_align: np.ndarray,
    n_neighbors: np.ndarray,
    a: float,
    J: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize forces by neighbor count (parallelized)."""
    n_fish = F_attr.shape[1]
    for i in prange(n_fish):
        n = max(n_neighbors[i], 1)
        inv_n = 1.0 / n
        F_attr[0, i] = a * F_attr[0, i] * inv_n
        F_attr[1, i] = a * F_attr[1, i] * inv_n
        F_align[0, i] = J * F_align[0, i] * inv_n
        F_align[1, i] = J * F_align[1, i] * inv_n
    return F_attr, F_align


def compute_neighbor_forces(
    X: np.ndarray,
    V: np.ndarray,
    fov_radius: float,
    a: float,
    Ra: float,
    J: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attraction and alignment forces using Numba.

    Args:
        X: Positions (2, N)
        V: Velocities (2, N)
        fov_radius: Neighbor search radius
        a: Attraction strength
        Ra: Repulsion radius
        J: Alignment strength

    Returns:
        (F_attraction, F_alignment) each (2, N)
    """
    n_fish = X.shape[1]
    X = np.ascontiguousarray(X, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)

    # KD-tree neighbor search (still use scipy - it's fast)
    tree = cKDTree(X.T)
    pairs = tree.query_pairs(fov_radius, output_type="ndarray")

    if len(pairs) == 0:
        zeros = np.zeros((2, n_fish), dtype=np.float32)
        return zeros, zeros.copy()

    # Symmetric expansion
    i_idx = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    j_idx = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)

    # Numba-accelerated force computation
    F_attr, F_align, n_neighbors = _compute_neighbor_forces(
        X, V, i_idx, j_idx, a, Ra, J, n_fish
    )

    return _normalize_forces(F_attr, F_align, n_neighbors, a, J)
