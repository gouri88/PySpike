"""
neuron.py

Implementation of a single Leaky Integrate-and-Fire (LIF) neuron, built
from scratch using only NumPy. No spiking-neural-network frameworks used.

The LIF model integrates incoming current into a membrane potential that
leaks toward a resting value over time. When the potential crosses a
threshold, the neuron fires a spike and the potential resets.
"""

import numpy as np


class LIFNeuron:
    """A single Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms). Controls how quickly the membrane
        potential leaks back toward the resting potential.
    v_rest : float
        Resting membrane potential (mV).
    v_reset : float
        Potential the membrane is reset to immediately after a spike (mV).
    v_threshold : float
        Firing threshold (mV). Crossing this triggers a spike.
    r_m : float
        Membrane resistance (MOhm). Scales how much effect input current
        has on the membrane potential.
    refractory_period : float
        Absolute refractory period (ms) during which the neuron cannot
        fire again after a spike.
    dt : float
        Simulation time step (ms).
    """

    def __init__(
        self,
        tau_m: float = 10.0,
        v_rest: float = -65.0,
        v_reset: float = -70.0,
        v_threshold: float = -50.0,
        r_m: float = 10.0,
        refractory_period: float = 2.0,
        dt: float = 0.1,
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r_m = r_m
        self.refractory_period = refractory_period
        self.dt = dt

        self.v = v_rest
        self.refractory_time_left = 0.0
        self.spike_times = []

    def reset(self):
        """Reset the neuron to its resting state."""
        self.v = self.v_rest
        self.refractory_time_left = 0.0
        self.spike_times = []

    def step(self, i_ext: float, t: float) -> bool:
        """Advance the neuron by one time step.

        Parameters
        ----------
        i_ext : float
            External input current (nA) at this time step.
        t : float
            Current simulation time (ms), used to log spike times.

        Returns
        -------
        bool
            True if the neuron fired a spike on this step.
        """
        if self.refractory_time_left > 0:
            self.refractory_time_left -= self.dt
            self.v = self.v_reset
            return False

        dv = (-(self.v - self.v_rest) + self.r_m * i_ext) / self.tau_m
        self.v += dv * self.dt

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_time_left = self.refractory_period
            self.spike_times.append(t)
            return True

        return False

    def simulate(self, current_trace: np.ndarray) -> dict:
        """Run the neuron over a full input current trace.

        Parameters
        ----------
        current_trace : np.ndarray
            1D array of input current values (nA), one per time step.

        Returns
        -------
        dict
            Dictionary with 'time', 'voltage', 'spikes' (boolean array),
            and 'spike_times' (list of ms values).
        """
        self.reset()
        n_steps = len(current_trace)
        time = np.arange(n_steps) * self.dt
        voltage = np.zeros(n_steps)
        spikes = np.zeros(n_steps, dtype=bool)

        for idx in range(n_steps):
            fired = self.step(current_trace[idx], time[idx])
            voltage[idx] = self.v
            spikes[idx] = fired

        return {
            "time": time,
            "voltage": voltage,
            "spikes": spikes,
            "spike_times": list(self.spike_times),
        }
