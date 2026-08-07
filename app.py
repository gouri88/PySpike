"""
app.py

Streamlit application for PySpike: an interactive Leaky Integrate-and-Fire
(LIF) neuron simulator with membrane potential visualization, spike train
plots, firing rate analysis, an F-I curve, and a step-by-step animation
mode.
"""

import time

import numpy as np
import streamlit as st

from neuron import LIFNeuron
from simulation import constant_current, run_simulation, compute_firing_rate, time_to_first_spike
from plots import plot_membrane_potential, plot_spike_train, plot_fi_curve
from utils import format_ms, format_hz
from experiment import run_fi_sweep

st.set_page_config(page_title="PySpike — LIF Neuron Simulator", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PySpike")
st.caption("An interactive Leaky Integrate-and-Fire (LIF) neuron, built from scratch — no SNN frameworks used.")

# ---- Sidebar: parameters ----
st.sidebar.header("Neuron Parameters")
tau_m = st.sidebar.slider("Membrane time constant τ_m (ms)", 1.0, 50.0, 10.0, 0.5)
v_rest = st.sidebar.slider("Resting potential (mV)", -80.0, -50.0, -65.0, 1.0)
v_reset = st.sidebar.slider("Reset potential (mV)", -80.0, -50.0, -70.0, 1.0)
v_threshold = st.sidebar.slider("Threshold (mV)", -60.0, -30.0, -50.0, 1.0)
r_m = st.sidebar.slider("Membrane resistance R_m (MΩ)", 1.0, 20.0, 10.0, 0.5)
refractory_period = st.sidebar.slider("Refractory period (ms)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.header("Stimulus")
amplitude = st.sidebar.slider("Input current (nA)", 0.0, 5.0, 1.8, 0.1)
duration = st.sidebar.slider("Duration (ms)", 100.0, 2000.0, 500.0, 50.0)
dt = st.sidebar.select_slider("Time step dt (ms)", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)

animate = st.sidebar.checkbox("Animation mode (step-by-step voltage)", value=False)

neuron_params = dict(
    tau_m=tau_m,
    v_rest=v_rest,
    v_reset=v_reset,
    v_threshold=v_threshold,
    r_m=r_m,
    refractory_period=refractory_period,
    dt=dt,
)

trace = constant_current(amplitude, duration, dt)

# ---- Main simulation ----
if animate:
    st.subheader("Animated Membrane Potential")
    placeholder = st.empty()
    neuron = LIFNeuron(**neuron_params)
    n_steps = len(trace)
    time_arr = np.arange(n_steps) * dt
    voltage = np.full(n_steps, np.nan)
    spikes = np.zeros(n_steps, dtype=bool)

    step_size = max(1, n_steps // 300)  # keep animation reasonably fast
    for idx in range(n_steps):
        fired = neuron.step(trace[idx], time_arr[idx])
        voltage[idx] = neuron.v
        spikes[idx] = fired
        if idx % step_size == 0 or idx == n_steps - 1:
            fig = plot_membrane_potential(
                time_arr[: idx + 1], voltage[: idx + 1], spikes[: idx + 1], v_threshold, v_reset
            )
            placeholder.pyplot(fig)

    result = {"time": time_arr, "voltage": voltage, "spikes": spikes, "spike_times": neuron.spike_times}
else:
    neuron = LIFNeuron(**neuron_params)
    result = run_simulation(neuron, trace)

    st.subheader("Membrane Potential")
    fig = plot_membrane_potential(
        result["time"], result["voltage"], result["spikes"], v_threshold, v_reset
    )
    st.pyplot(fig)

# ---- Spike train ----
st.subheader("Spike Train")
st.pyplot(plot_spike_train(result["spike_times"], duration))

# ---- Stats ----
rate = compute_firing_rate(result["spike_times"], duration)
first_spike = time_to_first_spike(result["spike_times"])

col1, col2, col3 = st.columns(3)
col1.metric("Total spikes", len(result["spike_times"]))
col2.metric("Firing rate", format_hz(rate))
col3.metric("Time to first spike", format_ms(first_spike))

# ---- F-I curve ----
st.subheader("Frequency-Current (F-I) Curve")
st.caption("Firing rate as a function of constant input current, holding all other parameters fixed.")

if st.button("Run F-I sweep"):
    with st.spinner("Sweeping input current..."):
        currents = np.linspace(0.0, 5.0, 25)
        sweep_params = {k: v for k, v in neuron_params.items() if k != "dt"}
        sweep_params["dt"] = dt
        currents, rates = run_fi_sweep(sweep_params, currents, duration=500.0)
    st.pyplot(plot_fi_curve(currents, rates))

st.markdown("---")
st.caption("Built from scratch with NumPy and Matplotlib — no spiking neural network frameworks used.")
