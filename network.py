"""
network.py

A minimal experiment showing several LIF neurons chained together, where
each neuron's spikes act as pulses of input current for the next neuron
in the chain. This is a simple illustration of feedforward spike
propagation, not a full SNN framework.
"""

import numpy as np
from neuron import LIFNeuron


def run_chain(
    n_neurons: int,
    input_current: np.ndarray,
    dt: float = 0.1,
    pulse_amplitude: float = 20.0,
    neuron_params: dict | None = None,
) -> list[dict]:
    """Simulate a chain of LIF neurons.

    The first neuron receives `input_current` directly. Each subsequent
    neuron receives a current pulse of `pulse_amplitude` on every time
    step where the previous neuron in the chain spiked, and zero
    otherwise.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the chain.
    input_current : np.ndarray
        Current trace driving the first neuron.
    dt : float
        Simulation time step (ms), used to build each neuron.
    pulse_amplitude : float
        Current (nA) injected into a downstream neuron when its upstream
        neuron spikes.
    neuron_params : dict, optional
        Extra keyword arguments passed to every `LIFNeuron` in the chain.

    Returns
    -------
    list of dict
        One simulation result dict (as returned by `LIFNeuron.simulate`)
        per neuron in the chain, in order.
    """
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
