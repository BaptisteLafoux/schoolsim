import numpy as np

def initialize_positions_rectangle(
    n_fish: int, tank_size: tuple[int, int]
) -> np.ndarray:
    w, h = tank_size
    x = np.random.uniform(-w / 2, w / 2, n_fish)
    y = np.random.uniform(-h / 2, h / 2, n_fish)
    return np.c_[x, y]


def initialize_positions_circle(n_fish: int, tank_radius: int) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n_fish)
    r = np.random.uniform(0, tank_radius, n_fish)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.c_[x, y]


def initialize_velocities(n_fish: int, v_initial: float) -> np.ndarray:
    return np.random.normal(0, v_initial, (2, n_fish))


def initialize_positions(
    n_fish: int, tank_size: tuple[int, int] | int, tank_shape: str
) -> np.ndarray:
    match tank_shape:
        case "rectangle":
            if not isinstance(tank_size, tuple):
                raise TypeError(
                    f"Tank size must be a tuple of width and height for rectangle shape, got {type(tank_size)}"
                )
            return initialize_positions_rectangle(n_fish, tank_size)
        case "circle":
            if not isinstance(tank_size, int):
                raise TypeError(
                    f"Tank size must be the radius for circle shape, got {type(tank_size)}"
                )
            return initialize_positions_circle(n_fish, tank_size)
        case _:
            raise ValueError(f"Invalid tank shape: {tank_shape}")
