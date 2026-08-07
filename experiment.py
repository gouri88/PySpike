import numpy as np
from neuron import LIFNeuron
from simulation import constant_current, run_simulation, compute_firing_rate


def run_fi_sweep(
    neuron_params: dict,
    current_range: np.ndarray,
    duration: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:

    dt = neuron_params.get("dt", 0.1)
    rates = np.zeros(len(current_range))

    for idx, amplitude in enumerate(current_range):
        neuron = LIFNeuron(**neuron_params)
        trace = constant_current(amplitude, duration, dt)
        result = run_simulation(neuron, trace)
        rates[idx] = compute_firing_rate(result["spike_times"], duration)

    return current_range, rates
