from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
        f_flee = (
            calculator.get_flee_force(predator.position, params.fov_radius, params.flee_strength)
            if predator
            else np.zeros_like(f_attraction)
        )

        forces = {
            "f_attraction": f_attraction,
            "f_alignment": f_alignment,
            "f_noise": f_noise,
            "f_propulsion": f_propulsion,
            "f_wall": f_wall,
            "f_flee": f_flee,
        }

        total_force = sum(forces.values())

        self.positions, self.velocities = run_integration(
            params.integration_scheme,
            self.positions,
            self.velocities,
            total_force, # type: ignore[arg-type]
            params.dt,
        )

        # if self.is_out_of_bounds(params.tank_size):
        #     raise ValueError("School is out of bounds")

        return Snapshot(self.positions, self.velocities, forces)

    def is_out_of_bounds(self, tank_size: tuple[int, int] | int) -> bool:
        if isinstance(tank_size, tuple):
            W, H = cast(tuple[int, int], tank_size)
            return bool(np.any(np.abs(self.positions) > max(W, H) / 2))
        else:
            return bool(np.any(np.abs(self.positions) > tank_size / 2))