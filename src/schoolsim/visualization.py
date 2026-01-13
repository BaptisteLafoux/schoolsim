import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.axes import Axes
from tqdm import tqdm


def _extract_data(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and velocity arrays from dataset."""
    pos = np.stack([ds['positions'].sel(dim="x").values, ds['positions'].sel(dim="y").values])
    vel = np.stack([ds['velocities'].sel(dim="x").values, ds['velocities'].sel(dim="y").values])
    return pos, vel


def _get_tank_limits(ds: xr.Dataset) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get x and y axis limits from tank shape."""
    match ds.attrs["tank_shape"]:
        case "rectangle":
            W, H = float(ds.attrs["tank_size"][0]), float(ds.attrs["tank_size"][1])
            return (-W / 2, W / 2), (-H / 2, H / 2)
        case "circle":
            R = float(ds.attrs["tank_size"])
            return (-R, R), (-R, R)
        case _:
            raise ValueError(f"Invalid tank shape: {ds.attrs['tank_shape']}")


def _init_tails(ax: Axes, pos: np.ndarray, n_fish: int, tail_length: int = 15):
    """Initialize tail scatter plot."""
    tail_pts = np.tile(pos[:, 0, :][:, :, None], tail_length).reshape(2, -1)
    return ax.scatter(
        tail_pts[0], tail_pts[1],
        c=np.tile(np.arange(tail_length), n_fish),
        cmap='Blues', alpha=0.5, s=5, zorder=0,
    )


def _init_fov_circles(ax: Axes, pos: np.ndarray, n_fish: int, fov_radius: float) -> list[mpatches.Circle]:
    """Initialize field of view circles."""
    circles = []
    for i in range(n_fish):
        circle = mpatches.Circle(
            (pos[0, 0, i], pos[1, 0, i]), fov_radius,
            fill=False, alpha=0.2, color='gray',
        )
        ax.add_patch(circle)
        circles.append(circle)
    return circles


def make_trajectory_animation(
    ds: xr.Dataset,
    filename: str,
    draw_velocity: bool = False,
    draw_fov: bool = False,
    acceleration_factor: int = 5,
    tail_length: int = 15,
) -> None:
    """Generate and save an animation from a simulation Dataset."""
    pos, vel = _extract_data(ds)
    x_lim, y_lim = _get_tank_limits(ds)
    
    n_fish = ds.attrs["n_fish"]
    num_steps = ds.attrs["num_steps"]
    dt = ds.attrs["dt"]
    n_frames = num_steps // acceleration_factor

    # Setup figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Initialize artists
    tails = _init_tails(ax, pos, n_fish, tail_length)
    (heads,) = ax.plot(pos[0, 0, :], pos[1, 0, :], 'ko', markersize=4, zorder=5)
    title = ax.set_title(f"t = 0.00 s | {n_fish} fish")

    quiver = None
    if draw_velocity:
        quiver = ax.quiver(
            pos[0, 0, :], pos[1, 0, :], vel[0, 0, :], vel[1, 0, :],
            angles='xy', scale_units='xy', scale=2, alpha=0.5, zorder=3
        )

    fov_circles = _init_fov_circles(ax, pos, n_fish, ds.attrs["fov_radius"]) if draw_fov else []

    def update(frame: int) -> list:
        t = min(frame * acceleration_factor, num_steps - 1)

        # Update positions
        heads.set_data(pos[0, t, :], pos[1, t, :])

        # Update tails
        t_start = max(0, t - tail_length * acceleration_factor)
        indices = np.linspace(t_start, t, tail_length, dtype=int)
        tails.set_offsets(pos[:, indices, :].reshape(2, -1).T)

        # Update optional elements
        if quiver is not None:
            quiver.set_offsets(pos[:, t, :].T)
            quiver.set_UVC(vel[0, t, :], vel[1, t, :])

        for i, circle in enumerate(fov_circles):
            circle.set_center((pos[0, t, i], pos[1, t, i]))

        title.set_text(f"t = {t * dt:.2f} s | {n_fish} fish")
        return [heads, tails, title] + ([quiver] if quiver else []) + fov_circles

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=dt * 1000 * acceleration_factor, blit=True)

    with tqdm(total=n_frames, desc="Saving animation", unit=" frames") as pbar:
        anim.save(filename, writer='ffmpeg', fps=30, progress_callback=lambda *_: pbar.update(1))
    
    plt.close(fig)
    print(f"Animation saved to {filename}")
