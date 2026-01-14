from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .forces import ForcesCalculator
from .initializer import initialize_positions, initialize_velocities
from .integration import run_integration
from .recorder import Snapshot

if TYPE_CHECKING:
    from .simulation import SimulationParameters
    from .predator import Predator


class School:
    def __init__(self, n_fish: int):
        self.n_fish = n_fish

    def initialize_in_bounds(
        self, tank_shape: str, tank_size: tuple[int, int] | int, v_initial: float = 1.0
    ) -> None:
        initial_positions = initialize_positions(self.n_fish, tank_size, tank_shape)
        initial_velocities = initialize_velocities(self.n_fish, v_initial)
        self.positions: np.ndarray = initial_positions
        self.velocities: np.ndarray = initial_velocities

    def update(self, params: SimulationParameters, predator: Predator | None = None) -> Snapshot:
        calculator = ForcesCalculator(
            self.positions, self.velocities, params.fov_radius
        )

        f_attraction = calculator.get_attraction_force(params.a, params.Ra)
        f_alignment = calculator.get_alignment_force(params.J)
        f_noise = calculator.get_noise_force(params.epsilon, self.n_fish)
        f_propulsion = calculator.get_propulsion_force(params.tau, params.v0)
        f_wall = calculator.get_wall_force(
            params.tank_shape, params.tank_size, params.delta, params.gamma_wall
        )

        total_force = f_attraction + f_alignment + f_noise + f_propulsion + f_wall
        
        if predator:
            f_flee = calculator.get_flee_force(predator.position, params.fov_radius, params.flee_strength)
            total_force += f_flee

        self.positions, self.velocities = run_integration(
            params.integration_scheme,
            self.positions,
            self.velocities,
            total_force,
            params.dt,
        )

        return Snapshot(
            self.positions,
            self.velocities,
            f_attraction,
            f_alignment,
            f_noise,
            f_propulsion,
            f_wall,
            f_flee if predator else None, # type: ignore
        )
