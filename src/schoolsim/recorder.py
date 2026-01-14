from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr  # type: ignore

if TYPE_CHECKING:
    from .simulation import SimulationParameters

@dataclass(slots=True)
class Snapshot:
    positions: np.ndarray
    velocities: np.ndarray
    f_attraction: np.ndarray
    f_alignment: np.ndarray
    f_noise: np.ndarray
    f_propulsion: np.ndarray
    f_wall: np.ndarray
    f_flee: np.ndarray | None

@dataclass(slots=True)
class PredatorSnapshot:
    position: np.ndarray
    velocity: np.ndarray

TimedSnapshot = tuple[float, Snapshot, PredatorSnapshot | None]

class Recorder:
    def __init__(self) -> None:
        self.snapshots: list[TimedSnapshot] = []
        self.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record(self, timestamp: float, snapshot: Snapshot, predator_snapshot: PredatorSnapshot | None = None) -> None:
        self.snapshots.append((timestamp, snapshot, predator_snapshot))

    def save_to_netcdf(self, filename: str, params: SimulationParameters) -> xr.Dataset:
        """Convert recorded snapshots to an xarray Dataset.

        Args:
            filename: The path to the output NetCDF file.
            metadata: A dictionary of metadata to add to the Dataset.

        Returns:
            The xarray Dataset.
        """
        if len(self.snapshots) == 0:
            raise ValueError("No snapshots recorded")

        data = [snapshot[1] for snapshot in self.snapshots]
        timestamps = [snapshot[0] for snapshot in self.snapshots]

        ds = xr.Dataset(
            data_vars={
                "positions": (
                    ("time", "dim", "fish"),
                    [snapshot.positions for snapshot in data],
                ),
                "velocities": (
                    ("time", "dim", "fish"),
                    [snapshot.velocities for snapshot in data],
                ),
                "f_attraction": (
                    ("time", "dim", "fish"),
                    [snapshot.f_attraction for snapshot in data],
                ),
                "f_alignment": (
                    ("time", "dim", "fish"),
                    [snapshot.f_alignment for snapshot in data],
                ),
                "f_noise": (
                    ("time", "dim", "fish"),
                    [snapshot.f_noise for snapshot in data],
                ),
                "f_propulsion": (
                    ("time", "dim", "fish"),
                    [snapshot.f_propulsion for snapshot in data],
                ),
                "f_wall": (
                    ("time", "dim", "fish"),
                    [snapshot.f_wall for snapshot in data],
                ),
            },
            coords={
                "time": np.array(timestamps),
                "dim": ["x", "y"],  # shape[0] = 2
                "fish": np.arange(params.n_fish),  # shape[1] = N
            },
            attrs={
                "creation_date": self.creation_date,
                **params.__dict__,
            },
        )

        ds.to_netcdf(filename)

        return ds
