from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from tqdm import tqdm

from .recorder import PredatorSnapshot, Recorder, Snapshot


def render_movie(
    recorder: Recorder,
    filename: str,
    *,
    fps: int = 24,
    dpi: int = 120,
) -> None:
    """Render an animation of the recorded school motion.

    The animation will be saved as a GIF (default) or MP4 if the extension matches.
    """
    snapshots = recorder.snapshots
    if not snapshots:
        raise ValueError("Recorder has no snapshots to render.")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_predator = any(predator_snapshot for _, _, predator_snapshot in snapshots)

    x_coords = np.hstack([snapshot.positions[0] for _, snapshot, _ in snapshots])
    y_coords = np.hstack([snapshot.positions[1] for _, snapshot, _ in snapshots])

    if has_predator:
        predator_positions = np.hstack(
            [
                predator_snapshot.position[:, None]
                for _, _, predator_snapshot in snapshots
                if predator_snapshot is not None
            ]
        )
        x_coords = np.hstack((x_coords, predator_positions[0]))
        y_coords = np.hstack((y_coords, predator_positions[1]))

    padding = max(np.ptp(x_coords), np.ptp(y_coords), 1.0) * 0.1
    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))

    fig, ax = plt.subplots()
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fish_scatter = ax.scatter([], [], c="tab:blue", marker="o", s=40) # type: ignore[arg-type]
    predator_scatter = (
        ax.scatter([], [], c="tab:red", marker="o", s=100) # type: ignore[arg-type]
        if has_predator
        else None
    )
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init() -> Sequence[plt.Artist]: # type: ignore[reportUnknownReturnType]
        fish_scatter.set_offsets(np.empty((0, 2)))
        if predator_scatter:
            predator_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        artists = [fish_scatter, time_text]
        if predator_scatter:
            artists.insert(1, predator_scatter)
        return artists

    def update(snapshot_entry: tuple[float, Snapshot, PredatorSnapshot | None]) -> Sequence[plt.Artist]: # type: ignore[reportUnknownReturnType]
        timestamp, snapshot, predator_snapshot = snapshot_entry
        fish_scatter.set_offsets(snapshot.positions.T)
        artists: list[plt.Artist] = [fish_scatter] # type: ignore[reportUnknownReturnType]
        if predator_scatter:
            if predator_snapshot:
                predator_coords = predator_snapshot.position.reshape(1, 2)
            else:
                predator_coords = np.empty((0, 2))
            predator_scatter.set_offsets(predator_coords)
            artists.append(predator_scatter)
        time_text.set_text(f"{timestamp:.2f}s")
        artists.append(time_text)
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=snapshots,
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    writer = FFMpegWriter(fps=fps) if path.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    with tqdm(total=len(snapshots), desc="Rendering", unit="frame") as pbar:
        animation.save(
            str(path),
            writer=writer, # type: ignore[arg-type]
            dpi=dpi,
            progress_callback=lambda i, n: pbar.update(1),  # type: ignore[arg-type]
        )
    plt.close(fig)
