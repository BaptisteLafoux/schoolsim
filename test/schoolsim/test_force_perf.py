import logging
from time import perf_counter

import numpy as np
import pytest

from schoolsim.forces import ForcesCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_FISH = 50


@pytest.fixture(scope='module')
def pos() -> np.ndarray:
    """200 fish randomly positioned in a 10x10 tank"""
    np.random.seed(42)
    return np.random.uniform(-5, 5, (2, N_FISH))


@pytest.fixture(scope='module')
def vel() -> np.ndarray:
    """200 fish with random velocities"""
    np.random.seed(43)
    return np.random.uniform(-1, 1, (2, N_FISH))


@pytest.fixture(scope='module')
def calc(pos: np.ndarray, vel: np.ndarray) -> ForcesCalculator:
    return ForcesCalculator(pos, vel, fov_radius=2.0)


def log_perf(name: str, duration: float):
    logger.info(f"{name}: {duration*1000:.3f} ms ({N_FISH} fish)")

class TestForcesPerf:
    def test_noise_force_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_noise_force(epsilon=0.1, n_fish=N_FISH)
        log_perf("get_noise_force", perf_counter() - start)

    def test_propulsion_force_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_propulsion_force(tau=1.0, v0=2.0)
        log_perf("get_propulsion_force", perf_counter() - start)

    def test_alignment_force_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_alignment_force(J=1.0)
        log_perf("get_alignment_force", perf_counter() - start)

    def test_attraction_force_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_attraction_force(a=1.0, Ra=0.5)
        log_perf("get_attraction_force", perf_counter() - start)

    def test_wall_force_rectangle_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_wall_force_rectangle(tank_size=(10, 10), delta=1.0, gammawall=1.0)
        log_perf("get_wall_force_rectangle", perf_counter() - start)

    def test_wall_force_circle_perf(self, calc: ForcesCalculator):
        start = perf_counter()
        _ = calc.get_wall_force_circle(radius=5.0, delta=1.0, gammawall=1.0)
        log_perf("get_wall_force_circle", perf_counter() - start)


class TestFullStepPerf:
    def test_full_force_computation(self, pos: np.ndarray, vel: np.ndarray):
        """Time a complete force computation cycle (fresh calculator)"""
        start = perf_counter()
        calc = ForcesCalculator(pos, vel, fov_radius=2.0)
        _ = calc.get_noise_force(epsilon=0.1, n_fish=N_FISH)
        _ = calc.get_propulsion_force(tau=1.0, v0=2.0)
        _ = calc.get_alignment_force(J=1.0)
        _ = calc.get_attraction_force(a=1.0, Ra=0.5)
        _ = calc.get_wall_force_rectangle(tank_size=(10, 10), delta=1.0, gammawall=1.0)
        log_perf("Full force computation", perf_counter() - start)
        logging.info(f"Number of steps per second: {1 / (perf_counter() - start):.1f}")
