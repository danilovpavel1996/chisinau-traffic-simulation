"""
run_sumo.py
-----------
Launches a full-day SUMO simulation for the Airport↔Botanica corridor.
Supports two modes:
  1. batch  — standard subprocess call (fast, no live control)
  2. traci  — connects TraCI for adaptive signal control

Usage
-----
    # Batch (baseline fixed-time):
    python src/simulation/run_sumo.py --config src/config/scenario.yaml

    # TraCI (adaptive controller):
    python src/simulation/run_sumo.py --config src/config/scenario.yaml \\
           --mode traci --controller adaptive_pressure
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── SUMO .sumocfg generator ───────────────────────────────────────────────────

def write_sumocfg(cfg: dict, cfg_out: str, add_file: str) -> None:
    """Generate a SUMO configuration XML file."""
    net      = os.path.abspath(cfg["network"]["net_file"])
    rte      = os.path.abspath(cfg["demand"]["routes_file"])
    out      = os.path.abspath(cfg["outputs"]["dir"])
    os.makedirs(out, exist_ok=True)
    tripinfo = os.path.join(out, "tripinfo.xml")
    add      = os.path.abspath(add_file)

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{rte}"/>
        <additional-files value="{add}"/>
    </input>
    <time>
        <begin value="{cfg['simulation']['begin']}"/>
        <end   value="{cfg['simulation']['end']}"/>
        <step-length value="{cfg['simulation']['step_length']}"/>
    </time>
    <output>
        <tripinfo-output value="{tripinfo}"/>
        <tripinfo-output.write-unfinished value="true"/>
    </output>
    <processing>
        <ignore-route-errors value="true"/>
        <no-step-log value="false"/>
        <time-to-teleport value="300"/>
    </processing>
    <report>
        <verbose value="true"/>
        <no-warnings value="false"/>
        <duration-log.statistics value="true"/>
    </report>
    <random>
        <seed value="{cfg['simulation']['seed']}"/>
    </random>
</configuration>
"""
    os.makedirs(os.path.dirname(cfg_out) or ".", exist_ok=True)
    with open(cfg_out, "w") as f:
        f.write(content)
    print(f"[run_sumo] sumocfg written → {cfg_out}")

# ── Edge-data additional file ────────────────────────────────────────────────

def write_edgedata_add(cfg: dict, add_out: str) -> None:
    add = os.path.abspath(add_out)
    edgedata = os.path.abspath(os.path.join(cfg["outputs"]["dir"], "edgedata.xml"))
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <edgeData id="ed_all" freq="900" file="{edgedata}"
              excludeEmpty="true"
              speedThreshold="-1"/>
</additional>
"""
    with open(add_out, "w") as f:
        f.write(content)
    print(f"[run_sumo] edgedata additional → {add_out}")


# ── Batch run ────────────────────────────────────────────────────────────────

def run_batch(sumocfg: str, gui: bool = False) -> None:
    binary = "sumo-gui" if gui else "sumo"
    cmd = [binary, "-c", sumocfg]
    print(f"[run_sumo] Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"SUMO exited with code {result.returncode}")
    print("[run_sumo] Simulation complete.")


# ── TraCI run ────────────────────────────────────────────────────────────────

def run_traci(sumocfg: str, controller_name: str, cfg: dict) -> None:
    try:
        import traci
    except ImportError:
        print("[run_sumo] traci not available — install via: pip install traci")
        sys.exit(1)

    port = 8813
    sumo_cmd = ["sumo", "-c", sumocfg]
    traci.start(sumo_cmd, port=port)
    print(f"[run_sumo] TraCI connected, controller={controller_name}")

    if controller_name == "adaptive_pressure":
        from controllers.adaptive_pressure import AdaptivePressureController
        controller = AdaptivePressureController(
            intersection_ids=[s["id"] for s in cfg["signals"]["pilot_intersections"]],
            cfg=cfg["signals"]["adaptive"],
        )
    elif controller_name == "fixed_time":
        controller = None   # SUMO handles it natively
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    step = 0
    total_steps = cfg["simulation"]["end"]

    while step < total_steps:
        traci.simulationStep()
        if controller:
            controller.step(step)
        step += 1
        if step % 3600 == 0:
            print(f"  [TraCI] simtime = {step//3600:02d}:00")

    traci.close()
    print("[run_sumo] TraCI simulation complete.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run SUMO corridor simulation")
    parser.add_argument("--config", default="src/config/scenario.yaml")
    parser.add_argument("--mode",   choices=["batch", "traci"], default="batch")
    parser.add_argument("--controller", default="adaptive_pressure",
                        help="Controller name (traci mode only)")
    parser.add_argument("--gui",    action="store_true",
                        help="Open SUMO-GUI (batch mode only)")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    cfg_dir  = cfg["outputs"]["dir"]
    sumocfg  = os.path.join(cfg_dir, "corridor.sumocfg")
    add_file = os.path.abspath(os.path.join(cfg["outputs"]["dir"], "edgedata.add.xml"))

    write_edgedata_add(cfg, add_file)
    # Inject the edgedata additional into the config
    write_sumocfg(cfg, sumocfg, add_file)

    if args.mode == "batch":
        run_batch(sumocfg, gui=args.gui)
    else:
        run_traci(sumocfg, args.controller, cfg)


if __name__ == "__main__":
    main()
