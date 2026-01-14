from .school import School
from .recorder import Recorder
from dataclasses import dataclass
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
    _validate_params(params)
    school = School(n_fish=params.n_fish)
    school.initialize_in_bounds(params.tank_shape, params.tank_size, params.v_initial)
    
    if params.predator:
        predator = Predator()
        predator.initialize_in_bounds(params.tank_shape, params.tank_size, params.predator_v_initial)
    else:
        predator = None
    
    predator_state = None
    recorder = Recorder()
    for step in range(params.num_steps):
        state = school.update(params, predator)
        
        if predator:
            predator_state = predator.update(school.positions, params)

        recorder.record(timestamp=step*params.dt, snapshot=state, predator_snapshot=predator_state if predator else None)
    return recorder