"""
Test the properties of the ForcesCalculator class (dense matrix version).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from schoolsim.forces import ForcesCalculator


@pytest.fixture(scope='module')
def pos() -> np.ndarray:
    """3 fish positions: shape (2, 3)"""
    return np.array([
        [0.0, 1.0, 0.0],  # x
        [0.0, 0.0, 1.0],  # y
    ])


@pytest.fixture(scope='module')
def vel() -> np.ndarray:
    """3 fish velocities: shape (2, 3)"""
    return np.array([
        [1.0, 0.0, -1.0],  # vx
        [0.0, 1.0,  0.0],  # vy
    ])


@pytest.fixture(scope='module')
def calc(pos: np.ndarray, vel: np.ndarray) -> ForcesCalculator:
    return ForcesCalculator(pos, vel, fov_radius=2.0)


class TestXij:
    def test_shape(self, calc: ForcesCalculator):
        """Xij should be (2, N, N)"""
        assert calc.Xij.shape == (2, 3, 3)

    def test_diagonal_is_zero(self, calc: ForcesCalculator):
        """X_ii = 0 (self-difference)"""
        for i in range(3):
            assert_allclose(calc.Xij[:, i, i], [0.0, 0.0])

    def test_antisymmetric(self, calc: ForcesCalculator):
        """X_ij = -X_ji"""
        for i in range(3):
            for j in range(3):
                assert_allclose(calc.Xij[:, i, j], -calc.Xij[:, j, i])


class TestRij:
    def test_shape(self, calc: ForcesCalculator):
        """rij should be (N, N)"""
        assert calc.rij.shape == (3, 3)

    def test_diagonal_is_one(self, calc: ForcesCalculator):
        """Diagonal is 1 (to avoid division by zero)"""
        for i in range(3):
            assert calc.rij[i, i] == 1.0

    def test_symmetric(self, calc: ForcesCalculator):
        """r_ij = r_ji"""
        assert_allclose(calc.rij, calc.rij.T)

    def test_known_distances(self, calc: ForcesCalculator):
        """Check expected distances"""
        # fish 0 at (0,0), fish 1 at (1,0) -> distance = 1
        assert_allclose(calc.rij[0, 1], 1.0)
        # fish 0 at (0,0), fish 2 at (0,1) -> distance = 1
        assert_allclose(calc.rij[0, 2], 1.0)
        # fish 1 at (1,0), fish 2 at (0,1) -> distance = sqrt(2)
        assert_allclose(calc.rij[1, 2], np.sqrt(2))


class TestVij:
    def test_shape(self, calc: ForcesCalculator):
        """Vij should be (2, N, N)"""
        assert calc.Vij.shape == (2, 3, 3)

    def test_diagonal_is_zero(self, calc: ForcesCalculator):
        """V_ii = 0 (self-difference)"""
        for i in range(3):
            assert_allclose(calc.Vij[:, i, i], [0.0, 0.0])


class TestFov:
    def test_shape(self, calc: ForcesCalculator):
        """fov should be (N, N)"""
        assert calc.fov.shape == (3, 3)

    def test_all_visible(self, calc: ForcesCalculator):
        """All fish within fov_radius=2, max dist is sqrt(2)"""
        assert calc.fov.sum() == 9  # all 1s

    def test_far_apart(self):
        """Fish far apart should not see each other"""
        pos = np.array([[0.0, 100.0], [0.0, 0.0]])
        vel = np.array([[0.0, 0.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        # Only diagonal should be 1 (self is always in fov due to rij[i,i]=1 < fov_radius)
        assert calc.fov[0, 1] == 0
        assert calc.fov[1, 0] == 0


class TestVnorm:
    def test_shape(self, calc: ForcesCalculator):
        assert calc.Vnorm.shape == (3,)

    def test_values(self, calc: ForcesCalculator):
        # vel = [[1,0,-1], [0,1,0]] -> norms = [1, 1, 1]
        assert_allclose(calc.Vnorm, [1.0, 1.0, 1.0])


class TestNInFov:
    def test_shape(self, calc: ForcesCalculator):
        assert calc.n_in_fov.shape == (3,)

    def test_counts(self, calc: ForcesCalculator):
        # All 3 fish visible to each, but n_in_fov excludes self -> 2 neighbors each
        assert_allclose(calc.n_in_fov, [2, 2, 2])

    def test_no_neighbors(self):
        """Fish with no neighbors should have n_in_fov = 1 (to avoid div by 0)"""
        pos = np.array([[0.0, 100.0], [0.0, 0.0]])
        vel = np.array([[0.0, 0.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        # No neighbors (only self in fov), so n_in_fov = max(1-1, 1) = 1
        assert_allclose(calc.n_in_fov, [1, 1])


class TestNonSelfMask:
    def test_shape(self, calc: ForcesCalculator):
        assert calc._non_self_mask.shape == (3, 3)

    def test_diagonal_is_zero(self, calc: ForcesCalculator):
        for i in range(3):
            assert calc._non_self_mask[i, i] == 0

    def test_off_diagonal_is_one(self, calc: ForcesCalculator):
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert calc._non_self_mask[i, j] == 1
