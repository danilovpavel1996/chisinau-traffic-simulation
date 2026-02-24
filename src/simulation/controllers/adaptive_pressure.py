"""
adaptive_pressure.py
--------------------
Max-pressure adaptive signal controller (TraCI-based).

Algorithm (per intersection, per control step)
----------------------------------------------
1. For each phase P, compute "pressure":
        pressure(P) = queue_in(P) - queue_out(P)
   where:
        queue_in  = sum of waiting vehicles on all incoming lanes served by P
        queue_out = sum of waiting vehicles on all outgoing lanes served by P

2. If current phase has been green for at least min_green:
        if pressure(current_phase) < max(pressure across all phases):
            switch to highest-pressure phase

3. Never exceed max_green for any phase.

Notes
-----
- Control is applied every `control_interval` simulation seconds (default 5).
- Lane-to-phase mapping is read from TraCI on startup.
"""

from __future__ import annotations
import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # traci imported at runtime


class AdaptivePressureController:
    def __init__(
        self,
        intersection_ids: list[str],
        cfg: dict,
        control_interval: int = 5,
    ):
        self.intersection_ids   = intersection_ids
        self.min_green          = cfg.get("min_green", 25)
        self.max_green          = cfg.get("max_green", 60)
        self.pressure_threshold = cfg.get("pressure_threshold", 3)
        self.control_interval   = control_interval

        # Runtime state per intersection
        self._phase_start:    dict[str, int] = {}
        self._current_phase:  dict[str, int] = {}
        self._phase_lanes:    dict[str, dict[int, list[str]]] = {}

        self._initialized = False

    # ── Initialization (called on first step after TraCI is up) ──────────────

    def _initialize(self) -> None:
        import traci
        for jid in self.intersection_ids:
            try:
                n_phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(jid)
                n = len(n_phases[0].phases) if n_phases else 4
            except Exception:
                n = 4
            self._current_phase[jid]  = 0
            self._phase_start[jid]    = 0
            self._phase_lanes[jid]    = self._build_phase_lane_map(jid, n)
        self._initialized = True
        print(f"[pressure_ctrl] Initialized for {len(self.intersection_ids)} intersections")

    def _build_phase_lane_map(self, jid: str, n_phases: int) -> dict[int, list[str]]:
        """Map phase index → list of incoming lane IDs served (stub, refined from link state)."""
        import traci
        try:
            links = traci.trafficlight.getControlledLinks(jid)
        except Exception:
            return {i: [] for i in range(n_phases)}

        # links: list of (incoming_lane, outgoing_lane, internal_lane) per link
        n = max(n_phases, 1)
        phase_map: dict[int, list[str]] = {i: [] for i in range(n)}
        for idx, link_list in enumerate(links):
            phase_idx = idx % n
            for incoming, _, _ in link_list:
                if incoming not in phase_map[phase_idx]:
                    phase_map[phase_idx].append(incoming)
        return phase_map

    # ── Per-step control ─────────────────────────────────────────────────────

    def step(self, sim_time: int) -> None:
        import traci

        if not self._initialized:
            self._initialize()

        if sim_time % self.control_interval != 0:
            return

        for jid in self.intersection_ids:
            self._control_intersection(jid, sim_time)

    def _control_intersection(self, jid: str, sim_time: int) -> None:
        import traci

        current_phase = self._current_phase.get(jid, 0)
        phase_start   = self._phase_start.get(jid, 0)
        green_duration = sim_time - phase_start

        # Enforce minimum green
        if green_duration < self.min_green:
            return

        # Enforce maximum green (force switch)
        if green_duration >= self.max_green:
            next_phase = (current_phase + 1) % len(self._phase_lanes[jid])
            self._switch_phase(jid, next_phase, sim_time)
            return

        # Compute pressure for all phases
        pressures = {}
        for phase_idx, lanes in self._phase_lanes[jid].items():
            q_in  = sum(traci.lane.getLastStepHaltingNumber(l)
                        for l in lanes if self._lane_exists(l))
            pressures[phase_idx] = q_in

        if not pressures:
            return

        best_phase    = max(pressures, key=pressures.get)
        best_pressure = pressures[best_phase]
        curr_pressure = pressures.get(current_phase, 0)

        # Switch if another phase has significantly higher pressure
        if (best_phase != current_phase and
                best_pressure - curr_pressure >= self.pressure_threshold):
            self._switch_phase(jid, best_phase, sim_time)

    def _switch_phase(self, jid: str, phase: int, sim_time: int) -> None:
        import traci
        try:
            traci.trafficlight.setPhase(jid, phase)
        except Exception as e:
            print(f"  [pressure_ctrl] WARNING: {jid} phase switch failed: {e}")
            return
        self._current_phase[jid] = phase
        self._phase_start[jid]   = sim_time

    @staticmethod
    def _lane_exists(lane_id: str) -> bool:
        import traci
        try:
            traci.lane.getLastStepHaltingNumber(lane_id)
            return True
        except Exception:
            return False
