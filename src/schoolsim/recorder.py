from dataclasses import dataclass
import numpy as np

@dataclass
class Snapshot:
    positions: np.ndarray
    velocities: np.ndarray
    f_attraction: np.ndarray
    f_alignment: np.ndarray
    f_noise: np.ndarray
    f_propulsion: np.ndarray

class Recorder:
    def __init__(self):
        self.snapshots: list[tuple[float, Snapshot]] = []

    def record(self, snapshot: Snapshot, timestamp: float) -> None:
        self.snapshots.append((timestamp, snapshot))