from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm

from .recorder import Recorder
from .school import School


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

def run_simulation(params: SimulationParameters, progress: bool = True) -> Recorder:
    """Run a single simulation with given parameters."""
    school = School(n_fish=params.n_fish)
    school.initialize_in_bounds(params.tank_shape, params.tank_size, params.v_initial)
    recorder = Recorder()
    
    steps = range(params.num_steps) if not progress else tqdm(range(params.num_steps), desc="Running simulation", unit=" steps")
    for step in steps:
        state = school.make_step(params)
        recorder.record(snapshot=state, timestamp=step * params.dt) # type: ignore
    return recorder


def run_batch(
    params_list: list[SimulationParameters],
    max_workers: int | None = None,
) -> list[Recorder]:
    """Run multiple simulations in parallel using processes.
    
    Args:
        params_list: List of simulation parameters to run.
        max_workers: Max processes. Defaults to cpu_count.
    
    Returns:
        List of Recorders in the same order as params_list.
    """
    results: dict[int, Recorder] = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_simulation, params, progress=False): i
            for i, params in enumerate(params_list)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batch simulations", unit=" sim"):
            idx = futures[future]
            results[idx] = future.result()
    
    return [results[i] for i in range(len(params_list))]

if __name__ == "__main__":
    from dataclasses import replace
    
    base_params = SimulationParameters(
        n_fish=50,
        num_steps=1000,
        dt=0.01,
        fov_radius=10.0,
        tank_shape="rectangle",
        tank_size=(2, 2),
        integration_scheme="symplectic_euler",
        v0=1.0,
        v_initial=0.5,
        a=1.0,
        Ra=0.5,
        J=1.0,
        epsilon=0.1,
        tau=1.0,
        delta=1.0,
        gamma_wall=1.0,
    )
    
    # Test batch with different fish counts
    params_list = [replace(base_params, n_fish=int(n)) for n in np.arange(10, 100, 2)]
    import os
    cpu_count: int = os.cpu_count() if os.cpu_count() is not None else 1 # type: ignore
    recorders = run_batch(params_list, max_workers=cpu_count - 1)
    
    for i, rec in enumerate(recorders):
        print(f"Simulation {i}: {len(rec.snapshots)} snapshots, {params_list[i].n_fish} fish")