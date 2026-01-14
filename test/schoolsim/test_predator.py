"""Tests for Predator class and flee force."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from schoolsim.predator import Predator
from schoolsim.forces import ForcesCalculator


class TestPredatorUpdate:
    @pytest.fixture
    def predator(self) -> Predator:
        p = Predator()
        p.position = np.array([0.0, 0.0])  # (2,)
        p.velocity = np.array([1.0, 0.0])  # (2,)
        return p

    @pytest.fixture
    def params(self) -> SimpleNamespace:
        return SimpleNamespace(
            integration_scheme="euler",
            dt=1.0,
            fov_radius=10.0,
            tank_shape="rectangle",
            tank_size=(100, 100),
            delta=1.0,
            gamma_wall=1.0,
        )

    def test_moves_toward_centroid(self, predator: Predator, params: SimpleNamespace):
        """Predator should move toward center of prey."""
        prey_positions = np.array([[8.0, 10.0, 12.0], [0.0, 0.0, 0.0]])
        
        predator.update(prey_positions, params)  # type: ignore[arg-type]
        
        assert predator.position[0] > 0

    def test_stationary_when_at_centroid(self, predator: Predator, params: SimpleNamespace):
        """No direction change when predator is at prey centroid."""
        prey_positions = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        params.dt = 0.1
        
        initial_pos = predator.position.copy()
        predator.update(prey_positions, params)  # type: ignore[arg-type]
        
        assert np.linalg.norm(predator.position - initial_pos) < 0.2


class TestFleeForce:
    @pytest.fixture
    def calc(self) -> ForcesCalculator:
        """3 fish in a line at x=0,5,10"""
        pos = np.array([[0.0, 5.0, 10.0], [0.0, 0.0, 0.0]])
        vel = np.zeros((2, 3))
        return ForcesCalculator(pos, vel, fov_radius=20.0)

    def test_shape(self, calc: ForcesCalculator):
        predator_pos = np.array([0.0, 5.0])
        F = calc.get_flee_force(predator_pos, flee_radius=10.0, flee_strength=1.0)
        assert F.shape == (2, 3)

    def test_flee_direction_away_from_predator(self, calc: ForcesCalculator):
        """Fish should flee away from predator."""
        predator_pos = np.array([0.0, 0.0])  # Predator at origin
        F = calc.get_flee_force(predator_pos, flee_radius=20.0, flee_strength=1.0)
        
        # Fish at (5,0) should flee in +x direction
        assert F[0, 1] > 0  # positive x component
        # Fish at (10,0) should also flee in +x direction
        assert F[0, 2] > 0

    def test_no_force_outside_radius(self, calc: ForcesCalculator):
        """Fish outside flee_radius should not flee."""
        predator_pos = np.array([0.0, 0.0])
        F = calc.get_flee_force(predator_pos, flee_radius=3.0, flee_strength=1.0)
        
        # Fish at (5,0) and (10,0) are outside radius=3
        assert_allclose(F[:, 1], [0.0, 0.0])
        assert_allclose(F[:, 2], [0.0, 0.0])

    def test_stronger_when_closer(self, calc: ForcesCalculator):
        """Closer fish should have stronger flee force."""
        predator_pos = np.array([-1.0, 0.0])  # Left of all fish
        F = calc.get_flee_force(predator_pos, flee_radius=20.0, flee_strength=1.0)
        
        # Fish at x=0 is closest, should have largest force magnitude
        force_fish0 = np.linalg.norm(F[:, 0])
        force_fish1 = np.linalg.norm(F[:, 1])
        force_fish2 = np.linalg.norm(F[:, 2])
        
        assert force_fish0 > force_fish1 > force_fish2
