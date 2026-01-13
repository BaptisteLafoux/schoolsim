from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

Record = tuple[float, 'Snapshot']

@dataclass(slots=True)
class Snapshot:
    positions: np.ndarray
    velocities: np.ndarray
    f_attraction: np.ndarray
    f_alignment: np.ndarray
    f_noise: np.ndarray
    f_propulsion: np.ndarray
    f_wall: np.ndarray
    
class Recorder:
    def __init__(self) -> None:
        self.snapshots: list[Record] = []
        self.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record(self, snapshot: Snapshot, timestamp: float) -> None:
        self.snapshots.append((timestamp, snapshot))
        
    def save_to_netcdf(self, filename: str, metadata: dict[str, Any]) -> xr.Dataset:
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
                "positions": (("time", "dim", "fish"), [snapshot.positions for snapshot in data]),
                "velocities": (("time", "dim", "fish"), [snapshot.velocities for snapshot in data]),
                "f_attraction": (("time", "dim", "fish"), [snapshot.f_attraction for snapshot in data]),
                "f_alignment": (("time", "dim", "fish"), [snapshot.f_alignment for snapshot in data]),
                "f_noise": (("time", "dim", "fish"), [snapshot.f_noise for snapshot in data]),
                "f_propulsion": (("time", "dim", "fish"), [snapshot.f_propulsion for snapshot in data]),
                "f_wall": (("time", "dim", "fish"), [snapshot.f_wall for snapshot in data]),
            },
            coords={
                "time": np.array(timestamps),
                "dim": ["x", "y"],  # shape[0] = 2
                "fish": np.arange(metadata["n_fish"]),  # shape[1] = N
            },
            attrs={
                "creation_date": self.creation_date,
                **metadata,
            },
        )
        
        ds.to_netcdf(filename)
        
        return ds


