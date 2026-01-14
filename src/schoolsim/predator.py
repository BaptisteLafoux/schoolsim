from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from .forces import ForcesCalculator
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
        """Move toward the centroid of the fish school."""
        calc = ForcesCalculator(
            self.position[:, None], self.velocity[:, None], params.fov_radius
        )
        
        chase_force = calc.get_chase_force_toward_nearest_fish(prey_positions).flatten()
        wall_force = calc.get_wall_force(
            params.tank_shape, params.tank_size, params.delta, params.gamma_wall
        ).flatten()

        total_force = chase_force + wall_force

        self.position, self.velocity = run_integration(
            params.integration_scheme,
            self.position,
            self.velocity,
            total_force,
            params.dt,
        )
        
        if self.is_out_of_bounds(params.tank_size):
            raise ValueError("Predator is out of bounds")

        return PredatorSnapshot(self.position, self.velocity)
    
    def is_out_of_bounds(self, tank_size: tuple[int, int] | int) -> bool:
        if isinstance(tank_size, tuple):
            W, H = cast(tuple[int, int], tank_size)
            return bool(np.any(np.abs(self.position) > max(W, H) / 2))
        else:
            return bool(np.any(np.abs(self.position) > tank_size / 2))