import numpy as np
import pytest
from numpy.testing import assert_allclose

from schoolsim.forces import ForcesCalculator


@pytest.fixture(scope='module')
def pos() -> np.ndarray:
    """4 fish: center, right, top, near-wall"""
    return np.array([
        [0.0, 2.0, 0.0, 4.5],  # x
        [0.0, 0.0, 2.0, 0.0],  # y
    ])


@pytest.fixture(scope='module')
def vel() -> np.ndarray:
    """4 fish velocities"""
    return np.array([
        [1.0, 1.0, 0.01, 1.2],  # vx (fish 3 moving toward wall)
        [0.0, 0.0, 1.02, 0.0],  # vy
    ])


@pytest.fixture(scope='module')
def calc(pos: np.ndarray, vel: np.ndarray) -> ForcesCalculator:
    return ForcesCalculator(pos, vel, fov_radius=3.0)


class TestNoiseForce:
    def test_shape(self, calc: ForcesCalculator):
        F = calc.get_noise_force(epsilon=0.1, n_fish=4)
        assert F.shape == (2, 4)

    def test_stats(self):
        """Noise should have mean ~0 and std ~epsilon"""
        calc = ForcesCalculator(np.zeros((2, 1000)), np.zeros((2, 1000)), 1.0)
        F = calc.get_noise_force(epsilon=0.5, n_fish=1000)
        assert abs(F.mean()) < 0.1
        assert abs(F.std() - 0.5) < 0.1


class TestPropulsionForce:
    def test_shape(self, calc: ForcesCalculator):
        F = calc.get_propulsion_force(tau=1.0, v0=2.0)
        assert F.shape == (2, 4)

    def test_accelerates_slow_fish(self):
        """Fish slower than v0 should accelerate in direction of motion"""
        pos = np.array([[0.0], [0.0]])
        vel = np.array([[0.5], [0.0]])  # slow fish moving right
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_propulsion_force(tau=1.0, v0=2.0)
        assert F[0, 0] > 0  # force in +x direction

    def test_decelerates_fast_fish(self):
        """Fish faster than v0 should decelerate"""
        pos = np.array([[0.0], [0.0]])
        vel = np.array([[3.0], [0.0]])  # fast fish
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_propulsion_force(tau=1.0, v0=2.0)
        assert F[0, 0] < 0  # force opposes motion


class TestAlignmentForce:
    def test_shape(self, calc: ForcesCalculator):
        F = calc.get_alignment_force(J=1.0)
        assert F.shape == (2, 4)

    def test_aligns_with_neighbors(self):
        """Fish should align with neighbors' velocity"""
        # Fish 0 stationary, fish 1 moving right
        pos = np.array([[0.0, 1.0], [0.0, 0.0]])
        vel = np.array([[0.0, 1.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=2.0)
        F = calc.get_alignment_force(J=1.0)
        # Fish 0 should feel force toward fish 1's velocity direction (Vj - Vi)
        assert F[0, 0] > 0  # pulled toward +x (fish 1's direction)


class TestAttractionForce:
    def test_shape(self, calc: ForcesCalculator):
        F = calc.get_attraction_force(a=1.0, Ra=0.5)
        assert F.shape == (2, 4)

    def test_attraction_at_distance(self):
        """Fish far apart should attract"""
        pos = np.array([[0.0, 3.0, 6.0], [0.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=5.0)
        F = calc.get_attraction_force(a=1.0, Ra=0.5)
        assert F[0, 0] > 0

    def test_repulsion_when_close(self):
        """Fish very close should repel"""
        # All 3 fish within fov so n_in_fov > 0 for all
        pos = np.array([[0.0, 0.3, 0.6], [0.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=2.0)
        F = calc.get_attraction_force(a=1.0, Ra=1.0)
        # Fish 0 should be repelled from neighbors (they're closer than Ra)
        assert F[0, 0] < 0


class TestWallForceRectangle:
    def test_shape(self):
        pos = np.array([[0.0, 4.0], [0.0, 0.0]])
        vel = np.array([[1.0, 1.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_rectangle(tank_size=(10, 10), delta=2.0, gammawall=1.0)
        assert F.shape == (2, 2)

    def test_repels_near_wall(self):
        """Fish near wall moving toward it should be repelled"""
        pos = np.array([[4.5], [0.0]])  # near right wall (x=5)
        vel = np.array([[1.0], [0.0]])  # moving toward wall
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_rectangle(tank_size=(10, 10), delta=2.0, gammawall=1.0)
        assert F[0, 0] < 0  # repelled in -x direction

    def test_no_force_far_from_wall(self):
        """Fish far from walls should feel no force"""
        pos = np.array([[0.0], [0.0]])  # center
        vel = np.array([[1.0], [0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_rectangle(tank_size=(10, 10), delta=1.0, gammawall=1.0)
        assert_allclose(F, [[0.0], [0.0]], atol=1e-10)


class TestWallForceCircle:
    def test_shape(self):
        pos = np.array([[0.0, 4.0], [0.0, 0.0]])
        vel = np.array([[1.0, 1.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_circle(radius=5.0, delta=2.0, gammawall=1.0)
        assert F.shape == (2, 2)

    def test_repels_near_boundary(self):
        """Fish near circular boundary moving outward should be repelled"""
        pos = np.array([[4.5], [0.0]])  # near boundary (r=5)
        vel = np.array([[1.0], [0.0]])  # moving outward
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_circle(radius=5.0, delta=2.0, gammawall=1.0)
        assert F[0, 0] < 0  # repelled inward

    def test_no_force_at_center(self):
        """Fish at center should feel no force (far from boundary)"""
        pos = np.array([[0.1], [0.0]])  # near center (avoid div by zero)
        vel = np.array([[1.0], [0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        F = calc.get_wall_force_circle(radius=5.0, delta=1.0, gammawall=1.0)
        assert_allclose(F, [[0.0], [0.0]], atol=1e-10)
