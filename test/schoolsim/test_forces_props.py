"""
Test the sparse properties of the ForcesCalculator class.
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


class TestNeighborPairs:
    def test_returns_bidirectional_pairs(self, calc: ForcesCalculator):
        """Each pair (i,j) should have corresponding (j,i)"""
        i_idx, j_idx = calc._neighbor_pairs
        pairs_set = set(zip(i_idx.tolist(), j_idx.tolist()))
        for i, j in list(pairs_set):
            assert (j, i) in pairs_set

    def test_all_within_fov(self, calc: ForcesCalculator):
        """All distances should be < 2 (fov_radius), max is sqrt(2)"""
        i_idx, j_idx = calc._neighbor_pairs
        for i, j in zip(i_idx, j_idx):
            dist = np.linalg.norm(calc.X[:, i] - calc.X[:, j])
            assert dist < calc.fov_radius

    def test_correct_pair_count(self, calc: ForcesCalculator):
        """3 fish all within fov: 3 pairs × 2 directions = 6"""
        i_idx, _ = calc._neighbor_pairs
        assert len(i_idx) == 6


class TestSparseXij:
    def test_shape(self, calc: ForcesCalculator):
        """Should be (2, n_pairs)"""
        assert calc._sparse_Xij.shape[0] == 2
        assert calc._sparse_Xij.shape[1] == len(calc._neighbor_pairs[0])

    def test_values(self, calc: ForcesCalculator):
        """Xij should be X[j] - X[i] (direction from i to j) for each pair"""
        i_idx, j_idx = calc._neighbor_pairs
        for k, (i, j) in enumerate(zip(i_idx, j_idx)):
            expected = calc.X[:, j] - calc.X[:, i]
            assert_allclose(calc._sparse_Xij[:, k], expected)


class TestSparseRij:
    def test_all_positive(self, calc: ForcesCalculator):
        """All distances should be > 0 (no self-pairs)"""
        assert np.all(calc._sparse_rij > 0)

    def test_known_distances(self, calc: ForcesCalculator):
        """Check expected distances"""
        # fish 0 at (0,0), fish 1 at (1,0) -> distance = 1
        # fish 0 at (0,0), fish 2 at (0,1) -> distance = 1
        # fish 1 at (1,0), fish 2 at (0,1) -> distance = sqrt(2)
        distances = calc._sparse_rij
        assert_allclose(min(distances), 1.0, rtol=1e-5)
        assert_allclose(max(distances), np.sqrt(2), rtol=1e-5)


class TestSparseVij:
    def test_shape(self, calc: ForcesCalculator):
        """Should be (2, n_pairs)"""
        assert calc._sparse_Vij.shape[0] == 2
        assert calc._sparse_Vij.shape[1] == len(calc._neighbor_pairs[0])


class TestSparseNormalized:
    def test_unit_vectors_normalized(self, calc: ForcesCalculator):
        """Xij_unit should have norm 1"""
        Xij_unit, _ = calc._sparse_normalized
        norms = np.linalg.norm(Xij_unit, axis=0)
        assert_allclose(norms, 1.0, rtol=1e-5)


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
        # All 3 fish visible to each, 2 neighbors each
        assert_allclose(calc.n_in_fov, [2, 2, 2])

    def test_no_neighbors(self):
        """Fish with no neighbors should have n_in_fov = 1 (to avoid div by 0)"""
        pos = np.array([[0.0, 100.0], [0.0, 0.0]])  # far apart
        vel = np.array([[0.0, 0.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        assert_allclose(calc.n_in_fov, [1, 1])


class TestKDTree:
    def test_empty_pairs_with_small_radius(self):
        """No pairs when fish are far apart"""
        pos = np.array([[0.0, 100.0], [0.0, 0.0]])
        vel = np.array([[0.0, 0.0], [0.0, 0.0]])
        calc = ForcesCalculator(pos, vel, fov_radius=1.0)
        i_idx, j_idx = calc._neighbor_pairs
        assert len(i_idx) == 0
        assert len(j_idx) == 0


class TestFloat32:
    def test_positions_are_float32(self, calc: ForcesCalculator):
        assert calc.X.dtype == np.float32

    def test_velocities_are_float32(self, calc: ForcesCalculator):
        assert calc.V.dtype == np.float32

    def test_sparse_xij_is_float32(self, calc: ForcesCalculator):
        assert calc._sparse_Xij.dtype == np.float32
