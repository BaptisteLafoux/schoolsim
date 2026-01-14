from functools import cached_property
from typing import cast

import numpy as np

class ForcesCalculator:
    def __init__(self, X: np.ndarray, V: np.ndarray, fov_radius: float):
        """
        Initialize the ForcesCalculator class.

        Args:
            X: (2, N) array of positions
            V: (2, N) array of velocities
            fov_radius: float, radius of the field of view
        """
        self.X = X
        self.V = V
        self.fov_radius = fov_radius
    
    @cached_property
    def Xij(self) -> np.ndarray:
        """ X_ij = X_i - X_j : positions of each individual in relation to all other individuals -- shape (N, N, 2) """
        return self.X[:, None, :] - self.X[:, :, None]
    
    @cached_property
    def rij(self) -> np.ndarray:
        """ r_ij = ||X_i - X_j|| : distance between each individual and all other individuals -- shape (N, N) """
        r = np.linalg.norm(self.Xij, axis=0)
        # Fill diagonal with 1 to avoid division by zero (self-distance)
        np.fill_diagonal(r, 1.0)
        return r 
    
    @cached_property
    def Vnorm(self) -> np.ndarray:
        """ ||V_i|| : norm of the velocities -- shape = (N, ) """
        return np.linalg.norm(self.V, axis=0) 
    
    @cached_property
    def Vij(self) -> np.ndarray:
        """ V_ij = V_i - V_j : velocities of each individual in relation to all other individuals -- shape (N, N, 2) """
        return self.V[:, None, :] - self.V[:, :, None]

    @cached_property
    def thetaij(self) -> np.ndarray:
        """ theta_ij = arccos(V_i . (X_i - X_j) / (||V_i|| * ||X_i - X_j||)) : angle between direction and motion of individual i and the unit vector from i to j -- shape (N, N) """
        return np.abs(np.einsum("im,inm->mn", self.V / self.Vnorm, -self.Xij / self.rij))

    @cached_property
    def fov(self) -> np.ndarray:
        """ fov_ij = (r_ij < R) : field of view of each individual -- shape (N, N) """
        return (self.rij < self.fov_radius).astype(int)
    
    @cached_property
    def n_in_fov(self) -> np.ndarray:
        """ n_in_fov_i = sum(fov_ij) - 1 : number of neighbors in the field of view (excluding self) -- shape (N, ) """
        # Subtract 1 to exclude self (diagonal is always in fov since rij[i,i] = 0 < fov_radius)
        # Use maximum to avoid 0 (for division safety)
        return np.maximum(self.fov.sum(axis=1) - 1, 1)
    
    def get_noise_force(self, epsilon: float, n_fish: int):
        return np.random.normal(0, epsilon, (2, n_fish))

    def get_propulsion_force(self, tau: float, v0: float):
        return (1 / tau) * (1 - self.Vnorm**2 / v0**2) * self.V

    def get_alignment_force(self, J: float):
        """This function will calculate the alignment force for each individual:

            F_ali = J * \sum_{j=1}^N (V_i - V_j) * Z_ali/N_i,al
            
            Here Z_ali/N_i,al is encompassed in the `fov` matrix 
            
        Returns:
            `np.array`: Alignment force -- shape (2, N)
        """
        return J * (self.Vij * self.fov / self.n_in_fov).sum(axis=2) # axis=2 sums over j
    
    @cached_property
    def _non_self_mask(self) -> np.ndarray:
        """Mask that is 0 on diagonal, 1 elsewhere -- shape (N, N)"""
        n = self.X.shape[1]
        return 1 - np.eye(n)

    def get_attraction_force(self, a: float, Ra: float):
        """This function will calculate the attraction-repulsion force for each individual:

            F_att = a * \sum_{j=1}^N [ xij / ||xij|| - Ra^2 * xij / ||xij||^3 ] * Z_att/N_i,att
            
            Here Z_att/N_i,att is encompassed in the `fov` matrix (/!\ here the fov is the same for alignment and attraction)
            
        Returns:
            `np.array`: Attraction force -- shape (2, N)
        """
        # Mask out self-interactions (diagonal)
        mask = self.fov * self._non_self_mask
        return a * ((self.Xij / self.rij - (Ra**2) * self.Xij / (self.rij**3)) * mask / self.n_in_fov).sum(axis=2)
    
    def get_wall_force_rectangle(self, tank_size: tuple[int, int], delta: float, gammawall: float):
        """ Wall repulsion force for rectangular tank centered at origin."""
        # Distances to walls: [North, South, East, West] for each fish
        # Tank spans from -W/2 to W/2 (x) and -H/2 to H/2 (y)
        W, H = tank_size
        distances = np.array([
            H / 2 - self.X[1],   # North
            H / 2 + self.X[1],   # South
            W / 2 - self.X[0],   # East
            W / 2 + self.X[0],   # West
        ])  # shape (4, N)
        
        # Normal vectors (pointing outward): [N, S, E, W]
        normals = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # shape (4, 2)
        
        # Repulsion strength: 1/d - 1/delta, zero if d > delta
        inv_delta = 1 / delta
        strength = np.maximum(1 / distances - inv_delta, 0)  # shape (4, N)
        
        # Velocity component toward wall (positive if moving toward it)
        v_toward_wall = np.einsum('wd,dN->wN', normals, self.V)  # shape (4, N)
        v_toward_wall = np.maximum(v_toward_wall, 0)
        
        # Force per wall: -strength * normal * v_toward_wall
        # Broadcast: (4, N) * (4, 2, 1) -> sum over walls
        F = -np.einsum('wN,wd,wN->dN', strength, normals, v_toward_wall)
        
        return gammawall * F

    def get_wall_force_circle(self, radius: float, delta: float, gammawall: float):
        """Wall repulsion force for circular tank centered at origin."""
        # Distance from center
        r = np.linalg.norm(self.X, axis=0)  # shape (N,)
        
        # Distance to wall
        d_wall = radius - r  # shape (N,)
        
        # Radial unit vector (pointing outward from center)
        r_safe = np.maximum(r, 1e-10)  # Avoid division by zero for fish at center
        n_radial = self.X / r_safe  # shape (2, N)
        
        # Repulsion strength
        inv_delta = 1 / delta
        strength = np.maximum(1 / d_wall - inv_delta, 0)  # shape (N,)
        
        # Velocity component toward wall (radial outward)
        v_toward_wall = np.einsum('dN,dN->N', n_radial, self.V)  # shape (N,)
        v_toward_wall = np.maximum(v_toward_wall, 0)
        
        # Force: -strength * n_radial * v_toward_wall
        F = -strength * v_toward_wall * n_radial  # shape (2, N)
        
        return gammawall * F
    
    def get_wall_force(self, tank_shape: str, tank_size: tuple[int, int] | int, delta: float, gammawall: float):
        match tank_shape:
            case "rectangle":
                return self.get_wall_force_rectangle(cast(tuple[int, int], tank_size), delta, gammawall)
            case "circle":
                return self.get_wall_force_circle(cast(int, tank_size), delta, gammawall)
            case _:
                raise ValueError(f"Invalid tank shape: {tank_shape}")
            
    def get_flee_force(self, predator_pos: np.ndarray, flee_radius: float, flee_strength: float) -> np.ndarray:
        """Fish flee from predator if within range."""
        diff = self.X - predator_pos[:, None]  # (2, N) direction away from predator
        dist = np.linalg.norm(diff, axis=0)    # (N,)
        
        # Only flee if within radius
        in_range = (dist < flee_radius).astype(float)
        
        # Normalize direction (avoid div by zero)
        dist_safe = np.maximum(dist, 1e-10)
        direction = diff / dist_safe
        
        # Stronger flee when closer (inverse distance)
        strength = flee_strength * (1 / dist_safe) * in_range
        
        return direction * strength