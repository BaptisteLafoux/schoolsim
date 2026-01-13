"""
Integration schemes for updating positions and velocities.

All functions take (positions, velocities, forces, dt) and return (new_positions, new_velocities).
"""

import numpy as np
from typing import Literal

def euler(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler (1st order).
    
    x_new = x + v * dt
    v_new = v + a * dt
    """
    new_positions = positions + velocities * dt
    new_velocities = velocities + forces * dt
    return new_positions, new_velocities


def symplectic_euler(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Symplectic Euler (1st order, better energy conservation).
    
    v_new = v + a * dt
    x_new = x + v_new * dt  (uses updated velocity)
    """
    new_velocities = velocities + forces * dt
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


def run_integration(integration_type: Literal["euler", "symplectic_euler"], positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    match integration_type:
        case "euler":
            return euler(positions, velocities, forces, dt)
        case "symplectic_euler":
            return symplectic_euler(positions, velocities, forces, dt)
        case _:
            raise ValueError(f"Invalid integration type: {integration_type}")