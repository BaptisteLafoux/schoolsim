import click

from .simulation import SimulationParameters, run_simulation
from .visualization import render_movie


@click.group()
def ssim() -> None:
    """SchoolSim - Fish schooling simulation."""
    pass


@ssim.command()
@click.option("--n-fish", "-n", default=100, help="Number of fish")
@click.option("--steps", "-s", default=1000, help="Number of simulation steps")
@click.option("--dt", default=0.05, help="Time step")
@click.option("--fov-radius", default=2.0, help="Field of view radius")
@click.option("--tank-shape", type=click.Choice(["rectangle", "circle"]), default="rectangle")
@click.option("--tank-width", default=10, help="Tank width (rectangle) or radius (circle)")
@click.option("--tank-height", default=10, help="Tank height (rectangle only)")
@click.option("--integration", type=click.Choice(["euler", "symplectic_euler"]), default="symplectic_euler")
@click.option("--v0", default=1.0, help="Target velocity")
@click.option("--v-initial", default=1, help="Initial velocity")
@click.option("--a", default=1.0, help="Attraction strength")
@click.option("--Ra", default=0.5, help="Repulsion radius")
@click.option("--J", default=1.0, help="Alignment strength")
@click.option("--epsilon", default=0.1, help="Noise strength")
@click.option("--tau", default=1.0, help="Propulsion time constant")
@click.option("--delta", default=1.0, help="Wall interaction distance")
@click.option("--gamma-wall", default=1.0, help="Wall force strength")
@click.option("--predator/--no-predator", default=False, help="Enable predator")
@click.option("--predator-speed", default=1.5, help="Predator initial velocity")
@click.option("--flee-strength", default=2.0, help="Fish flee force strength")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--movie", type=click.Path(), help="Save an animation of the run (GIF or MP4).")
@click.option("--movie-fps", default=24, help="Frames per second for the animation.")
@click.option("--movie-dpi", default=120, help="Animation DPI.")
def run(
    n_fish: int,
    steps: int,
    dt: float,
    fov_radius: float,
    tank_shape: str,
    tank_width: int,
    tank_height: int,
    integration: str,
    v0: float,
    v_initial: float,
    a: float,
    ra: float,
    j: float,
    epsilon: float,
    tau: float,
    delta: float,
    gamma_wall: float,
    predator: bool,
    predator_speed: float,
    flee_strength: float,
    output: str | None,
    movie: str | None,
    movie_fps: int,
    movie_dpi: int,
) -> None:
    """Run a fish schooling simulation."""
    tank_size: tuple[int, int] | int = tank_width if tank_shape == "circle" else (tank_width, tank_height)

    params = SimulationParameters(
        n_fish=n_fish,
        num_steps=steps,
        dt=dt,
        fov_radius=fov_radius,
        tank_shape=tank_shape,  # type: ignore[arg-type]
        tank_size=tank_size,
        integration_scheme=integration,  # type: ignore[arg-type]
        v0=v0,
        v_initial=v_initial,
        a=a,
        Ra=ra,
        J=j,
        epsilon=epsilon,
        tau=tau,
        delta=delta,
        gamma_wall=gamma_wall,
        predator=predator,
        predator_v_initial=predator_speed,
        flee_strength=flee_strength,
    )

    click.echo(f"Running simulation with {n_fish} fish for {steps} steps...")
    recorder = run_simulation(params)
    click.echo(f"Completed! {len(recorder.snapshots)} snapshots recorded.")

    if output:
        click.echo(f"Saving dataset to {output}…")
        recorder.save_to_netcdf(output, params)

    if movie:
        click.echo(f"Rendering animation to {movie}…")
        render_movie(recorder, movie, fps=movie_fps, dpi=movie_dpi)


@ssim.command()
def info() -> None:
    """Show simulation info and defaults."""
    click.echo("SchoolSim - Fish Schooling Simulation")
    click.echo("=====================================")
    click.echo("Default parameters:")
    click.echo("  n_fish: 100")
    click.echo("  steps: 1000")
    click.echo("  dt: 0.01")
    click.echo("  integration: symplectic_euler")
    click.echo("  tank_shape: rectangle (20x20)")
    click.echo("  predator: disabled")
    click.echo("  predator_speed: 1.5")
    click.echo("  flee_strength: 2.0")