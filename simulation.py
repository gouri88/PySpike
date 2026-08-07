import numpy as np
from neuron import LIFNeuron


def constant_current(amplitude: float, duration: float, dt: float) -> np.ndarray:
    n_steps = int(duration / dt)
    return np.full(n_steps, amplitude, dtype=float)


def step_current(
    amplitude: float, duration: float, dt: float, onset: float, offset: float
) -> np.ndarray:
    """Generate a current that is zero, then steps to `amplitude` between
    `onset` and `offset` (ms), then returns to zero."""
    n_steps = int(duration / dt)
    trace = np.zeros(n_steps, dtype=float)
    onset_idx = int(onset / dt)
    offset_idx = int(offset / dt)
    trace[onset_idx:offset_idx] = amplitude
    return trace


def noisy_current(
    amplitude: float, duration: float, dt: float, noise_std: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Generate a constant current with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    base = np.full(n_steps, amplitude, dtype=float)
    noise = rng.normal(0, noise_std, n_steps)
    return base + noise


def run_simulation(neuron: LIFNeuron, current_trace: np.ndarray) -> dict:
    """Run a neuron over a given current trace and return the results."""
    return neuron.simulate(current_trace)


def compute_firing_rate(spike_times: list, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 0.0
    return len(spike_times) / (duration_ms / 1000.0)


def time_to_first_spike(spike_times: list) -> float | None:
    """Return the time of the first spike (ms), or None if no spike."""
    if not spike_times:
        return None
    return spike_times[0]
