from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .integration import run_integration
from .initializer import initialize_positions, initialize_velocities
from .recorder import PredatorSnapshot

if TYPE_CHECKING:
    from .simulation import SimulationParameters

class Predator:
    """A predator that chases the center of mass of a fish school."""
    def __init__(self) -> None:
        self.position: np.ndarray = np.array([])
        self.velocity: np.ndarray = np.array([])
    
    def initialize_in_bounds(
        self, tank_shape: str, tank_size: tuple[int, int] | int, v_initial: float = 1.0
    ) -> None:
        initial_position = initialize_positions(1, tank_size, tank_shape)
        initial_velocity = initialize_velocities(1, v_initial)
        self.position = initial_position.flatten()  # (2,) for single entity
        self.velocity = initial_velocity.flatten()  # (2,)

    def update(self, prey_positions: np.ndarray, params: SimulationParameters) -> PredatorSnapshot:
        """Move toward the centroid of the fish school.
        Args:
            fish_positions: Array of shape (2, N)
            dt: Time step
        """
        centroid = prey_positions.mean(axis=1)  # (2,)
        direction = centroid - self.position
        dist = np.linalg.norm(direction)
        if dist > 1e-10:
            direction = direction / dist
        self.position, self.velocity = run_integration(
            params.integration_scheme,
            self.position,
            self.velocity,
            self.velocity * direction,
            params.dt,
        )
        
        return PredatorSnapshot(self.position, self.velocity)