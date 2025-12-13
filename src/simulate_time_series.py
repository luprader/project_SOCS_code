"""Functions to simulate time series and get functional correlations."""

import numpy as np

from BaloonWindkessel import balloonWindkessel


def run_kuramoto(
    C: np.ndarray,
    distance_matrix: np.ndarray,
    dt: float,
    total_time: int,
    seed: int = 0,
    initial_phases: np.ndarray = None,
    noise_factor: float = 0.0,
    coupling_factor: float = 0.0,
) -> np.ndarray:
    """Classic, undelayed Kuramoto integrator (Euler) with white noise.

    C: (N,N) coupling matrix (2D). Diagonal is zeroed inside.
    distance_matrix: (N,N) distance matrix (2D).
    dt: timestep
    total_time: total simulation time (in seconds)
    seed: RNG seed for reproducibility
    initial_phases: initial phases of the oscillators (optional)
    noise_factor: scaling factor of white noise added to phase derivatives.
                  Noise is sqrt(dt)-scaled for physical correctness.
    coupling_factor: scaling factor for coupling term.

    Returns phases with shape (N, n_steps).
    """
    C = np.array(C, dtype=float)
    np.fill_diagonal(C, 0.0)
    N = C.shape[0]

    # normalise C rowsums
    row_sums = C.sum(axis=1, keepdims=True)
    C = C / row_sums
    C[np.isnan(C)] = 0.0

    n_steps = int(total_time / dt)
    warmup_steps = int(10 / dt)

    used_seed = seed if seed != 0 else np.random.SeedSequence().entropy
    rng = np.random.default_rng(used_seed)

    theta = rng.uniform(0, 2 * np.pi, N) if initial_phases is None else initial_phases

    # Model parameters according to Cabral et al. 2011
    f = rng.normal(60, 5, N)  # Hz
    noise = rng.normal(0, 3, (N, warmup_steps + n_steps))  # rad/s
    mean_delay = 11e-3  # 11ms
    delay_steps_base = mean_delay / dt  # convert to steps in t iteration

    omega = 2 * np.pi * f

    # Precompute integer delay steps once (in samples)
    delay_steps = np.rint(
        delay_steps_base * distance_matrix / np.mean(distance_matrix)
    ).astype(int)
    max_delay = int(np.max(delay_steps)) + 1

    # Circular buffer for warmup: only store max_delay + 1 time steps
    buffer_size = max_delay + 1
    phase_buffer = np.zeros((N, buffer_size))
    phase_buffer[:, 0] = theta

    # Warmup simulation (20 seconds) - use circular buffer
    for t in range(1, warmup_steps):
        # print(f"Warmup step {t}/{warmup_steps}", end="\r")
        buffer_idx = t % buffer_size
        prev_idx = (t - 1) % buffer_size

        # Clip delays so we never index before 0 and never beyond t-1
        delay_indices = np.clip(t - delay_steps, 0, t - 1).astype(int)

        # Map delay indices to buffer positions
        buffer_delay_indices = delay_indices % buffer_size

        # Get the phases at the delayed time points
        delayed_phases = np.take_along_axis(phase_buffer, buffer_delay_indices, axis=1)

        # Phase difference: source minus target
        phase_diff = delayed_phases - phase_buffer[:, prev_idx][:, None]  # (N, N)

        dtheta = (
            omega
            + coupling_factor * np.sum(C * np.sin(phase_diff), axis=1)
            + noise_factor * noise[:, t]
        )

        phase_buffer[:, buffer_idx] = (phase_buffer[:, prev_idx] + dt * dtheta) % (
            2 * np.pi
        )

    # Save phases for the actual simulation
    phases = np.full((N, n_steps), np.nan, dtype=np.float64)
    # Initialize with the last warmup phase
    last_warmup_idx = (warmup_steps - 1) % buffer_size
    phases[:, 0] = phase_buffer[:, last_warmup_idx]

    # Main simulation loop - save all phases
    for t in range(1, n_steps):
        actual_t = warmup_steps + t
        # print(f"Simulation step {t}/{n_steps}", end="\r")

        # Clip delays accounting for warmup period
        delay_indices = np.clip(actual_t - delay_steps, 0, actual_t - 1).astype(int)

        # Determine which phases come from buffer vs saved array
        delayed_phases = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                delay_t = delay_indices[i, j]
                if delay_t < warmup_steps:
                    # Phase is in the circular buffer
                    buffer_idx = delay_t % buffer_size
                    delayed_phases[i, j] = phase_buffer[j, buffer_idx]
                else:
                    # Phase is in the saved array
                    saved_idx = delay_t - warmup_steps
                    delayed_phases[i, j] = phases[j, saved_idx]

        # Phase difference: source minus target
        phase_diff = delayed_phases - phases[:, t - 1][:, None]  # (N, N)

        dtheta = (
            omega
            + coupling_factor * np.sum(C * np.sin(phase_diff), axis=1)
            + noise_factor * noise[:, warmup_steps + t]
        )

        phases[:, t] = (phases[:, t - 1] + dt * dtheta) % (2 * np.pi)
    return phases


def calculate_bold(
    time_series: np.ndarray,
    time_step: float,
    sample_rate: float,
) -> np.ndarray:
    """Calculate functional connectivity matrix from time series.

    time_series: (N, T) array of N time series with T time points each.
    time_step: time step between samples (in seconds).
    sample_rate: target sampling rate (less than time_step).

    Returns functional connectivity matrix of shape (N, N).
    """
    # issue with overflow with timeseries >= 500 seconds
    # bold, s, f, v, q = balloonWindkessel(time_series, time_step)
    bold = time_series.copy()

    # Downsample BOLD to desired sample rate
    downsample_factor = int(sample_rate / time_step)
    bold_downsampled = bold[:, ::downsample_factor]

    return bold_downsampled
