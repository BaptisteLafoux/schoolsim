from functools import cached_property
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

NORMALS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float32)


class ForcesCalculator:
    def __init__(self, X: np.ndarray, V: np.ndarray, fov_radius: float):
        """
        Initialize the ForcesCalculator class.

        Args:
            X: (2, N) array of positions
            V: (2, N) array of velocities
            fov_radius: float, radius of the field of view
        """
        self.X = X.astype(np.float32)
        self.V = V.astype(np.float32)
        self.fov_radius = fov_radius
        self.n_fish = X.shape[1]

    @cached_property
    def _kdtree(self) -> cKDTree:
        """KD-tree for efficient neighbor search"""
        return cKDTree(self.X.T)  # cKDTree expects (N, 2)
    
    @cached_property
    def _neighbor_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Sparse neighbor indices: (i_indices, j_indices) for pairs within fov_radius"""
        pairs = self._kdtree.query_pairs(self.fov_radius, output_type='ndarray')
        if len(pairs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        # Include both directions (i,j) and (j,i)
        return (
            np.concatenate([i_idx, j_idx]).astype(np.int32),
            np.concatenate([j_idx, i_idx]).astype(np.int32)
        )
    
    @cached_property
    def _sparse_Xij(self) -> np.ndarray:
        """Xij for neighbor pairs: X_j - X_i (direction from i to j), shape (2, n_pairs)"""
        i_idx, j_idx = self._neighbor_pairs
        if len(i_idx) == 0:
            return np.zeros((2, 0), dtype=np.float32)
        return self.X[:, j_idx] - self.X[:, i_idx]
    
    @cached_property
    def _sparse_rij(self) -> np.ndarray:
        """Distance for neighbor pairs: shape (n_pairs,)"""
        return np.linalg.norm(self._sparse_Xij, axis=0)
    
    @cached_property
    def _sparse_rij_safe(self) -> np.ndarray:
        """rij with minimum value to avoid division by zero"""
        return np.maximum(self._sparse_rij, 1e-10)

    @cached_property
    def _sparse_normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute Xij/r and Xij/r³ in one pass for efficiency"""
        r = self._sparse_rij_safe
        r3 = r ** 3
        Xij = self._sparse_Xij
        return Xij / r, Xij / r3

    @cached_property
    def _sparse_Vij(self) -> np.ndarray:
        """Vij for neighbor pairs: V_j - V_i (velocity difference), shape (2, n_pairs)"""
        i_idx, j_idx = self._neighbor_pairs
        if len(i_idx) == 0:
            return np.zeros((2, 0), dtype=np.float32)
        return self.V[:, j_idx] - self.V[:, i_idx]
    
    @cached_property
    def n_in_fov(self) -> np.ndarray:
        """Number of neighbors for each fish (excluding self): shape (N,)"""
        i_idx, _ = self._neighbor_pairs
        counts = np.bincount(i_idx, minlength=self.n_fish)
        return np.maximum(counts, 1)  # Avoid division by zero

    @cached_property
    def Vnorm(self) -> np.ndarray:
        """||V_i|| : norm of the velocities -- shape (N,)"""
        return np.linalg.norm(self.V, axis=0)
    
    def get_noise_force(self, epsilon: float, n_fish: int) -> np.ndarray:
        return np.random.normal(0, epsilon, (2, n_fish)).astype(np.float32)

    def get_propulsion_force(self, tau: float, v0: float) -> np.ndarray:
        return (1 / tau) * (1 - self.Vnorm**2 / v0**2) * self.V

    def get_alignment_force(self, J: float) -> np.ndarray:
        r"""Alignment force using sparse neighbor computation.

        F_ali = J * \sum_{j in neighbors} (V_i - V_j) / n_neighbors
            
        Returns:
            np.ndarray: Alignment force -- shape (2, N)
        """
        i_idx, _ = self._neighbor_pairs
        if len(i_idx) == 0:
            return np.zeros((2, self.n_fish), dtype=np.float32)
    
        # Sum Vij contributions per fish
        F = np.zeros((2, self.n_fish), dtype=np.float32)
        np.add.at(F.T, i_idx, self._sparse_Vij.T)

        return J * F / self.n_in_fov

    def get_attraction_force(self, a: float, Ra: float) -> np.ndarray:
        r"""Attraction-repulsion force using sparse neighbor computation.
            
        F_att = a * \sum_{j in neighbors} [ Xij/r - Ra^2 * Xij/r^3 ] / n_neighbors
            
        Returns:
            np.ndarray: Attraction force -- shape (2, N)
        """
        i_idx, _ = self._neighbor_pairs
        if len(i_idx) == 0:
            return np.zeros((2, self.n_fish), dtype=np.float32)
        
        Xij_unit, Xij_over_r3 = self._sparse_normalized
        force_per_pair = Xij_unit - (Ra**2) * Xij_over_r3
        
        # Sum contributions per fish
        F = np.zeros((2, self.n_fish), dtype=np.float32)
        np.add.at(F.T, i_idx, force_per_pair.T)
        
        return a * F / self.n_in_fov
    
    def get_wall_force_rectangle(self, tank_size: tuple[int, int], delta: float, gammawall: float) -> np.ndarray:
        """Wall repulsion force for rectangular tank centered at origin."""
        W, H = tank_size
        inv_delta = np.float32(1 / delta)
        
        distances = np.array([
            H / 2 - self.X[1],   # North
            H / 2 + self.X[1],   # South
            W / 2 - self.X[0],   # East
            W / 2 + self.X[0],   # West
        ], dtype=np.float32)
        
        strength = np.maximum(1 / distances - inv_delta, 0)
        v_toward_wall = np.maximum(np.einsum('wd,dN->wN', NORMALS, self.V), 0)
        F = -np.einsum('wN,wd,wN->dN', strength, NORMALS, v_toward_wall)
        
        return gammawall * F

    def get_wall_force_circle(self, radius: float, delta: float, gammawall: float) -> np.ndarray:
        """Wall repulsion force for circular tank centered at origin."""
        r = np.linalg.norm(self.X, axis=0)
        d_wall = radius - r
        
        r_safe = np.maximum(r, 1e-10)
        n_radial = self.X / r_safe
        
        inv_delta = np.float32(1 / delta)
        strength = np.maximum(1 / d_wall - inv_delta, 0)
        v_toward_wall = np.maximum(np.einsum('dN,dN->N', n_radial, self.V), 0)
        
        F = -strength * v_toward_wall * n_radial
        return gammawall * F
    
    def get_wall_force(self, tank_shape: str, tank_size: tuple[int, int] | int, delta: float, gammawall: float):
        match tank_shape:
            case "rectangle":
                return self.get_wall_force_rectangle(cast(tuple[int, int], tank_size), delta, gammawall)
            case "circle":
                return self.get_wall_force_circle(cast(int, tank_size), delta, gammawall)
            case _:
                raise ValueError(f"Invalid tank shape: {tank_shape}")

    def get_attraction_and_alignment_force(self, a: float, Ra: float, J: float) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated fused attraction + alignment computation."""
        from .forces_numba import compute_neighbor_forces
        return compute_neighbor_forces(self.X, self.V, self.fov_radius, a, Ra, J)
            
