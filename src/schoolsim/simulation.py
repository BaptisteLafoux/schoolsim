from .school import School
from .recorder import Recorder
from dataclasses import dataclass, field
from .predator import Predator
from typing import Literal
import numpy as np

@dataclass
class SimulationParameters:
    fov_radius: float
    a: float
    Ra: float
    J: float
    epsilon: float
    tau: float
    v0: float | np.ndarray
    integration_scheme: Literal["euler", "symplectic_euler"]
    tank_shape: Literal["rectangle", "circle"]
    tank_size: tuple[int, int] | int
    v_initial: float
    num_steps: int
    dt: float
    n_fish: int
    delta: float
    gamma_wall: float
    predator: bool
    predator_v_initial: float
    flee_strength: float
    
def _validate_params(params: SimulationParameters) -> None:
    if isinstance(params.v0, np.ndarray) and params.v0.shape != (params.n_fish,):
        raise ValueError(f"v0 must be a ({params.n_fish},) array")
    if params.n_fish <= 0:
        raise ValueError("n_fish must be greater than 0")
    if params.dt <= 0:
        raise ValueError("dt must be greater than 0")


def run_simulation(params: SimulationParameters) -> Recorder:
    return SimulationRunner(params).run()


@dataclass
class SimulationRunner:
    params: SimulationParameters
    school: School = field(init=False)
    predator: Predator | None = field(init=False)
    recorder: Recorder = field(init=False)

    def __post_init__(self) -> None:
        _validate_params(self.params)
        self.school = School(n_fish=self.params.n_fish)
        self.school.initialize_in_bounds(
            self.params.tank_shape, self.params.tank_size, self.params.v_initial
        )

        if self.params.predator:
            predator = Predator()
            predator.initialize_in_bounds(
                self.params.tank_shape, self.params.tank_size, self.params.predator_v_initial
            )
            self.predator = predator
        else:
            self.predator = None

        self.recorder = Recorder()

    def run(self) -> Recorder:
        for step in range(self.params.num_steps):
            snapshot = self.school.update(self.params, self.predator)
            predator_snapshot = None
            if self.predator:
                predator_snapshot = self.predator.update(self.school.positions, self.params)

            self.recorder.record(
                timestamp=step * self.params.dt,
                snapshot=snapshot,
                predator_snapshot=predator_snapshot,
            )

        return self.recorder