import numpy as np
from neuron import LIFNeuron


def run_chain(
    n_neurons: int,
    input_current: np.ndarray,
    dt: float = 0.1,
    pulse_amplitude: float = 20.0,
    neuron_params: dict | None = None,
) -> list[dict]:

    params = dict(neuron_params or {})
    params["dt"] = dt

    n_steps = len(input_current)
    results = []
    drive = input_current

    for _ in range(n_neurons):
        neuron = LIFNeuron(**params)
        result = neuron.simulate(drive)
        results.append(result)
        drive = np.where(result["spikes"], pulse_amplitude, 0.0)

    return results
