"""
fixed_time.py
-------------
Fixed-time signal controller (no-op — SUMO handles it natively).
This module exists so run_sumo.py can reference it uniformly.
"""


class FixedTimeController:
    """Pass-through: SUMO's built-in static TLS handles timing."""

    def __init__(self, *args, **kwargs):
        pass

    def step(self, sim_time: int) -> None:
        pass   # nothing to do
