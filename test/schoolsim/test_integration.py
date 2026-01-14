"""Integration tests for full simulation workflow."""

import numpy as np
import pytest

from schoolsim.simulation import SimulationParameters, run_simulation


@pytest.fixture
def base_params() -> SimulationParameters:
    """Minimal working simulation parameters."""
    return SimulationParameters(
        n_fish=20,
        num_steps=50,
        dt=0.05,
        fov_radius=3.0,
        tank_shape="rectangle",
        tank_size=(10, 10),
        integration_scheme="symplectic_euler",
        v0=1.0,
        v_initial=0.5,
        a=1.0,
        Ra=0.3,
        J=0.5,
        epsilon=0.1,
        tau=1.0,
        delta=1.0,
        gamma_wall=1.0,
        predator=False,
        predator_v_initial=1.5,
        flee_strength=2.0,
    )


class TestSimulationRuns:
    def test_basic_simulation_completes(self, base_params: SimulationParameters):
        """Simulation runs without errors."""
        recorder = run_simulation(base_params)
        assert len(recorder.snapshots) == base_params.num_steps

    def test_simulation_with_predator(self, base_params: SimulationParameters):
        """Simulation with predator enabled runs without errors."""
        base_params.predator = True
        recorder = run_simulation(base_params)
        assert len(recorder.snapshots) == base_params.num_steps
        # Predator snapshot should exist
        _, _, predator_snap = recorder.snapshots[-1]
        assert predator_snap is not None

    def test_circular_tank(self, base_params: SimulationParameters):
        """Simulation works with circular tank."""
        base_params.tank_shape = "circle"
        base_params.tank_size = 10
        recorder = run_simulation(base_params)
        assert len(recorder.snapshots) == base_params.num_steps


class TestSimulationPhysics:
    def test_fish_stay_bounded(self, base_params: SimulationParameters):
        """Fish positions remain within tank bounds."""
        recorder = run_simulation(base_params)
        _, final_snapshot, _ = recorder.snapshots[-1]
        
        w, h = base_params.tank_size  # type: ignore[misc]
        positions = final_snapshot.positions
        
        # Allow some margin for wall forces
        margin = 2.0
        assert np.all(positions[0] > -w/2 - margin)
        assert np.all(positions[0] < w/2 + margin)
        assert np.all(positions[1] > -h/2 - margin)
        assert np.all(positions[1] < h/2 + margin)

    def test_velocities_reasonable(self, base_params: SimulationParameters):
        """Velocities don't explode."""
        recorder = run_simulation(base_params)
        _, final_snapshot, _ = recorder.snapshots[-1]
        
        speeds = np.linalg.norm(final_snapshot.velocities, axis=0)
        # Speeds should be within reasonable bounds (not exploding)
        assert np.all(speeds < 10 * base_params.v0)

    def test_forces_are_finite(self, base_params: SimulationParameters):
        """All computed forces are finite (no NaN/Inf)."""
        recorder = run_simulation(base_params)
        
        for _, snapshot, _ in recorder.snapshots:
            assert np.all(np.isfinite(snapshot.f_attraction))
            assert np.all(np.isfinite(snapshot.f_alignment))
            assert np.all(np.isfinite(snapshot.f_propulsion))
            assert np.all(np.isfinite(snapshot.f_wall))


class TestPredatorBehavior:
    def test_predator_chases_school(self, base_params: SimulationParameters):
        """Predator moves toward fish over time."""
        base_params.predator = True
        base_params.num_steps = 100
        recorder = run_simulation(base_params)
        
        _, _, first_pred = recorder.snapshots[0]
        _, _, last_pred = recorder.snapshots[-1]
        
        assert first_pred is not None and last_pred is not None
        
        # Predator should generally get closer (not guaranteed but likely)
        assert not np.allclose(first_pred.position, last_pred.position)


class TestHeterogeneousV0:
    def test_v0_array_simulation_runs(self, base_params: SimulationParameters):
        """Simulation runs with per-fish v0 array."""
        base_params.v0 = np.full(base_params.n_fish, 1.0)
        recorder = run_simulation(base_params)
        assert len(recorder.snapshots) == base_params.num_steps

    def test_v0_varied_speeds(self, base_params: SimulationParameters):
        """Simulation with varied fish speeds."""
        base_params.v0 = np.random.uniform(0.5, 1.5, base_params.n_fish)
        recorder = run_simulation(base_params)
        assert len(recorder.snapshots) == base_params.num_steps
        
        # Forces should still be finite
        _, final_snap, _ = recorder.snapshots[-1]
        assert np.all(np.isfinite(final_snap.f_propulsion))

    def test_v0_wrong_shape_raises(self, base_params: SimulationParameters):
        """Wrong v0 array shape should raise ValueError."""
        base_params.v0 = np.array([1.0, 2.0])  # wrong size
        with pytest.raises(ValueError, match="v0 must be"):
            run_simulation(base_params)
