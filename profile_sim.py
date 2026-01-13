import cProfile
import pstats
from pstats import SortKey
from schoolsim.simulation import SimulationParameters, run_simulation
params = SimulationParameters(
    n_fish=500,  # Representative count
    num_steps=200,  # Enough to get stable measurements
    dt=0.01,
    fov_radius=5.0,
    tank_shape="rectangle",
    tank_size=(20, 20),
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
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_simulation(params, progress=False)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    
    print("\n=== TOP 20 BY CUMULATIVE TIME ===")
    stats.print_stats(20)
    
    print("\n=== TOP 20 BY TOTAL TIME (self) ===")
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)