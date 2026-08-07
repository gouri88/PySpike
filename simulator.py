"""
simulator.py

Thin compatibility layer re-exporting the simulation engine under an
alternate module name, so both `simulation` and `simulator` imports work
across the codebase.
"""

from simulation import (
    constant_current,
    step_current,
    noisy_current,
    run_simulation,
    compute_firing_rate,
    time_to_first_spike,
)

__all__ = [
    "constant_current",
    "step_current",
    "noisy_current",
    "run_simulation",
    "compute_firing_rate",
    "time_to_first_spike",
]
