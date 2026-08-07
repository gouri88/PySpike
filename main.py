from neuron import LIFNeuron
from simulation import constant_current, run_simulation, compute_firing_rate, time_to_first_spike


def main():
    duration = 500.0  # ms
    dt = 0.1  # ms
    amplitude = 1.8  # nA

    neuron = LIFNeuron(dt=dt)
    trace = constant_current(amplitude, duration, dt)
    result = run_simulation(neuron, trace)

    n_spikes = len(result["spike_times"])
    rate = compute_firing_rate(result["spike_times"], duration)
    first_spike = time_to_first_spike(result["spike_times"])

    print("PySpike — LIF Neuron Command-Line Demo")
    print("-" * 40)
    print(f"Input current:      {amplitude} nA")
    print(f"Duration:            {duration} ms")
    print(f"Total spikes:        {n_spikes}")
    print(f"Firing rate:         {rate:.2f} Hz")
    if first_spike is not None:
        print(f"Time to first spike: {first_spike:.2f} ms")
    else:
        print("Time to first spike: no spike occurred")


if __name__ == "__main__":
    main()
