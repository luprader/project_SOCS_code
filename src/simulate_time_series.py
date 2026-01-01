"""Functions to simulate time series and get functional correlations."""

import numpy as np

# from src.BaloonWindkessel import balloonWindkessel
def run_kuramoto(
    C: np.ndarray,
    distance_matrix: np.ndarray,
    dt: float,
    total_time: float,
    coupling_factor: float = 18.0,
    noise_factor: float = 1.0,
    mean_delay: float = 0.0,  # seconds; set to 0 to disable delays
    # seed: int = 0,
    initial_phases: np.ndarray | None = None,
) -> np.ndarray:
    """Delayed Kuramoto integrator (Euler) with uniform intrinsic frequencies.

    C: (N,N) coupling matrix (2D). Diagonal is zeroed inside.
    distance_matrix: (N,N) distance matrix (2D) for computing per-edge delays.
    dt: timestep (seconds).
    total_time: total simulation time (seconds).
    coupling_factor: global coupling strength multiplier.
    noise_factor: scaling factor of white noise added to phase derivatives (rad/s).
    mean_delay: mean propagation delay (seconds). Set to 0 to disable delays.
    seed: RNG seed for reproducibility.
    initial_phases: initial phases of the oscillators (optional, radians).

    Returns phases with shape (N, n_steps), in radians.
    """
    C = np.array(C, dtype=float)
    np.fill_diagonal(C, 0.0)
    N = C.shape[0]

    # Normalize C to mean 1
    C = C / np.mean(C[C > 0])

    n_steps = int(total_time / dt)

    # used_seed = seed if seed != 0 else np.random.SeedSequence().entropy
    rng = np.random.default_rng()

    # Parameters according to Cabral et al. 2011
    theta = (
        rng.uniform(0, 2 * np.pi, N)
        if initial_phases is None
        else np.asarray(initial_phases, dtype=float)
    )

    f = rng.normal(60, 5, N)  # Hz
    noise = rng.normal(0, 3, (N, n_steps)) # rad/s

    omega = 2 * np.pi * f

    # Noise array
    noise = (
        rng.normal(0, 3, (N, n_steps)) if noise_factor > 0 else np.zeros((N, n_steps))
    )

    # (Delays are disabled here; keep placeholder in case re-enabled)
    if mean_delay > 0 and np.mean(distance_matrix) > 0:
        delay_steps_base = mean_delay / dt
        delay_steps = np.rint(
            delay_steps_base * distance_matrix / np.mean(distance_matrix)
        ).astype(int)
    else:
        delay_steps = np.zeros_like(distance_matrix, dtype=int)



    phases = np.zeros((N, n_steps), dtype=float)
    phases[:, 0] = theta

    for t in range(1, n_steps):
        # Phase difference: theta_j - theta_i (correct sign for attractive coupling)
        if t-1 < np.max(delay_steps):
            clipped_delays = np.clip(t-1 - delay_steps, 0, t-1)
        else:
            clipped_delays = t-1 - delay_steps  # (N, N)

        delayed_phases = np.take_along_axis(phases, clipped_delays, axis=1)  # (N, N)

        phase_diff = delayed_phases - phases[:, t - 1].T  # (N, N)

        dtheta = (
            omega
            + coupling_factor * np.sum(C * np.sin(phase_diff), axis=1)
            + noise_factor * noise[:, t]
        )

        phases[:, t] = (phases[:, t - 1] + dt * dtheta) % (2 * np.pi)

    return phases


# def calculate_bold(
#     time_series: np.ndarray,
#     time_step: float,
#     sample_rate: float,
# ) -> np.ndarray:
#     """Calculate functional connectivity matrix from time series.

#     time_series: (N, T) array of N time series with T time points each.
#     time_step: time step between samples (in seconds).
#     sample_rate: target sampling rate (less than time_step).

#     Returns functional connectivity matrix of shape (N, N).
#     """
#     # issue with overflow with timeseries >= 500 seconds
#     bold, s, f, v, q = balloonWindkessel(time_series, time_step)
#     # bold = time_series.copy()

#     # Downsample BOLD to desired sample rate
#     downsample_factor = int(sample_rate / time_step)
#     bold_downsampled = bold[:, ::downsample_factor]

#     return bold_downsampled
