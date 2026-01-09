import numpy as np

def verlet(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> np.ndarray:
    return positions + velocities * dt + 0.5 * dt**2 * forces

def euler(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> np.ndarray:
    return positions + velocities * dt + forces * dt

def runge_kutta(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> np.ndarray:
    return positions + velocities * dt + 0.5 * dt**2 * forces

def runge_kutta_4(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray, dt: float) -> np.ndarray:
    return positions + velocities * dt + 0.5 * dt**2 * forces