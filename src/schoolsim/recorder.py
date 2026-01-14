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
    forces: dict[str, np.ndarray]

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
                **{
                    force_name: (
                        ("time", "dim", "fish"),
                        [snapshot.forces[force_name] for snapshot in data],
                    )
                    for force_name in data[0].forces
                },
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
