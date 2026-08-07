"""
plots.py

Matplotlib plotting helpers for visualizing LIF neuron simulations:
membrane potential traces with spike markers, spike raster/train plots,
and the Frequency-Current (F-I) curve.

All plots use a dark theme suited to an educational Streamlit interface.
"""

import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#0e1117"
GRID_COLOR = "#2a2e39"
LINE_COLOR = "#00d4ff"
SPIKE_COLOR = "#ff3b3b"
THRESHOLD_COLOR = "#ffb703"
TEXT_COLOR = "#e6e6e6"


def _style_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_membrane_potential(time, voltage, spikes, v_threshold, v_reset):
    """Plot the membrane potential trace with red markers at spike times
    and a dashed line for the firing threshold."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style_axes(ax)

    ax.plot(time, voltage, color=LINE_COLOR, linewidth=1.3, label="Membrane potential")
    ax.axhline(v_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1, label="Threshold")

    spike_times = time[spikes]
    if len(spike_times) > 0:
        ax.scatter(
            spike_times,
            [v_threshold] * len(spike_times),
            color=SPIKE_COLOR,
            zorder=5,
            s=30,
            label="Spike",
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title("Membrane Potential")
    ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COLOR, framealpha=0.3)
    fig.tight_layout()
    return fig


def plot_spike_train(spike_times, duration):
    """Plot a raster-style spike train."""
    fig, ax = plt.subplots(figsize=(9, 1.6), facecolor=DARK_BG)
    _style_axes(ax)

    if spike_times:
        ax.vlines(spike_times, 0, 1, color=SPIKE_COLOR, linewidth=1.5)

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.set_title("Spike Train")
    fig.tight_layout()
    return fig


def plot_fi_curve(currents, rates):
    """Plot the Frequency-Current (F-I) curve."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
    _style_axes(ax)

    ax.plot(currents, rates, color=LINE_COLOR, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Input current (nA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("F-I Curve")
    fig.tight_layout()
    return fig
