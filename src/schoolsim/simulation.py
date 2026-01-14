from .school import School
from .recorder import Recorder
from dataclasses import dataclass
from .predator import Predator
from typing import Literal

@dataclass
class SimulationParameters:
    fov_radius: float
    a: float
    Ra: float
    J: float
    epsilon: float
    tau: float
    v0: float
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

def run_simulation(params: SimulationParameters) -> Recorder:
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