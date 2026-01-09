from .school import School
from .recorder import Recorder, Snapshot
from dataclasses import dataclass

from typing import Literal, Optional

@dataclass
class SimulationParameters:
    fov_radius: float
    a: float
    Ra: float
    J: float
    epsilon: float
    tau: float
    v0: float
    integration_method: Literal["verlet", "euler", "runge_kutta", "runge_kutta_4"]
    tank_shape: Literal["rectangle", "circle"]
    tank_size: tuple[int, int] | int
    v_initial: float
    num_steps: int
    dt: float
    n_fish: int
    delta: float
    gamma_wall: float

def run_simulation(params: SimulationParameters) -> list[tuple[float, Snapshot]]:
    school = School(n_fish=params.n_fish)
    school.initialize_in_bounds(params.tank_shape, params.tank_size, params.v_initial)
    recorder = Recorder()
    for step in range(params.num_steps):
        state = school.make_step(params.dt, params)
        recorder.record(snapshot=state, timestamp=step*params.dt)
    return recorder.snapshots