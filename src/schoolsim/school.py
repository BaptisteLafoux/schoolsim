import numpy as np

from . import integration
from .forces import ForcesCalculator
from .initializer import initialize_positions, initialize_velocities
from .recorder import Snapshot
from .simulation import SimulationParameters


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

    def make_step(self, dt: float, params: SimulationParameters) -> Snapshot:
        calculator = ForcesCalculator(
            self.positions, self.velocities, params.fov_radius
        )

        f_attraction = calculator.get_attraction_force(params.a, params.Ra)
        f_alignment = calculator.get_alignment_force(params.J)
        f_noise = calculator.get_noise_force(params.epsilon, self.n_fish)
        f_propulsion = calculator.get_propulsion_force(params.tau, params.v0)
        f_wall = calculator.get_wall_force(params.tank_shape, params.tank_size, params.delta, params.gamma_wall)

        self.velocities = (
            self.velocities + (f_attraction + f_alignment + f_noise + f_propulsion + f_wall) * dt
        )

        self.positions = integration.verlet(
            self.positions,
            self.velocities,
            f_attraction + f_alignment + f_noise + f_propulsion,
            dt,
        )

        self.positions = self.positions + self.velocities * dt

        return Snapshot(
            positions=self.positions,
            velocities=self.velocities,
            f_attraction=f_attraction,
            f_alignment=f_alignment,
            f_noise=f_noise,
            f_propulsion=f_propulsion,
        )
