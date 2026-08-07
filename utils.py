from simulation import compute_firing_rate, time_to_first_spike

__all__ = ["compute_firing_rate", "time_to_first_spike", "format_ms", "format_hz"]


def format_ms(value) -> str:
    """Format a millisecond value for display, handling None."""
    if value is None:
        return "No spike"
    return f"{value:.2f} ms"


def format_hz(value: float) -> str:
    """Format a firing rate value for display."""
    return f"{value:.2f} Hz"
