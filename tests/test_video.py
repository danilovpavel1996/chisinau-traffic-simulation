"""Fast unit tests for the video pipeline (no video decoding, no downloads).

    .venv-video/bin/python -m pytest tests -q
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.video import drives
from src.video.localize import accuracy, systematic
from src.video.motion import stops, turns
from src.video.roads import Roads, wrap


def frames(yaw_deg_s, moving=True, dt=0.2):
    """Synthetic ego_states output: one row per sampled frame."""
    yaw = np.asarray(yaw_deg_s, dtype=float)
    t = np.arange(len(yaw)) * dt
    return pd.DataFrame(dict(t=t, yaw_deg_s=yaw, yaw_smooth=yaw, turning=np.abs(yaw) > 5.0,
                             moving=np.broadcast_to(moving, yaw.shape)))


def test_wrap_angles():
    assert wrap(190) == -170
    assert wrap(-190) == 170
    assert wrap(0) == 0
    np.testing.assert_allclose(wrap(np.array([360, 45, 225])), [0, 45, -135])


def test_right_turn_detected_with_net_angle():
    # 6 s at −15 deg/s (yaw < 0 = right) = 90° right, with straight driving around it
    e = frames([0] * 25 + [-15] * 30 + [0] * 25)
    t = turns(e)
    assert len(t) == 1
    assert t.direction.iat[0] == "right"
    assert t.angle_deg_approx.iat[0] == pytest.approx(90, abs=5)


def test_roundabout_wiggle_is_one_manoeuvre_with_net_angle():
    # right in, left around, right out — gaps < 3 s merge into one manoeuvre
    e = frames([0] * 10 + [-20] * 10 + [0] * 5 + [12] * 15 + [0] * 5 + [-20] * 10 + [0] * 10)
    t = turns(e)
    assert len(t) == 1
    net = -20 * 2 * 2 + 12 * 3       # deg: two 2-s right arcs, one 3-s left arc
    assert t.direction.iat[0] == "right"
    assert t.angle_deg_approx.iat[0] == pytest.approx(abs(net), abs=6)


def test_small_heading_change_is_not_a_turn():
    assert turns(frames([0] * 10 + [-8] * 10 + [0] * 10)).empty    # 16°, below MIN_TURN


def test_stops_with_and_without_red_light():
    e = frames([0] * 65, moving=np.r_[np.ones(10), np.zeros(25), np.ones(10), np.zeros(20)].astype(bool))
    e["light"] = None
    e.loc[12:20, "light"] = "red"
    s = stops(e)
    assert list(s.cause) == ["signal", "congestion_or_other"]
    assert s.duration_s.iat[0] == pytest.approx(4.8, abs=0.3)


def test_oneway_edges_get_a_contraflow_copy():
    G = nx.MultiDiGraph()
    G.add_node(1, x=28.80, y=47.00)
    G.add_node(2, x=28.81, y=47.00)
    G.add_node(3, x=28.81, y=47.01)
    G.add_edge(1, 2, 0, length=760.0, oneway=True)                       # one-way
    G.add_edge(2, 3, 0, length=1110.0, oneway=False)                     # two-way, both directions
    G.add_edge(3, 2, 0, length=1110.0, oneway=False,
               geometry=LineString([(28.81, 47.01), (28.81, 47.00)]))
    r = Roads(G)
    assert (2, 1, 0) in r.contra                       # reversed copy of the one-way edge
    assert (3, 2, 0) not in r.contra                   # real two-way edge is not contra-flow
    assert r.data((2, 1, 0)) is G.edges[1, 2, 0]       # attributes come from the OSM edge
    assert {v for v, *_ in r.out[2]} == {1, 3}


def test_systematic_resampling_keeps_heavy_particles():
    w = np.array([0.0, 0.0, 1.0, 0.0])
    idx = systematic(w, np.random.default_rng(0))
    assert set(idx) == {2}


def test_accuracy_of_the_truth_itself_is_within_metric_resolution():
    truth = [(47.00, 28.80), (47.00, 28.81), (47.01, 28.81)]
    lat = np.linspace(47.00, 47.00, 20).tolist() + np.linspace(47.00, 47.01, 20).tolist()
    lon = np.linspace(28.80, 28.81, 20).tolist() + [28.81] * 20
    a = accuracy(pd.DataFrame(dict(lat=lat, lon=lon)), truth)
    assert a["median_m"] <= 3 and a["coverage"] == "100%"     # truth sampled every 5 m → ≤2.5 m resolution


def test_registered_drive_loads():
    d = drives.load("2026_10_22_10_21AM")
    assert d["video"].name == "2026_10_22_10_21AM.MOV"
    assert d["folder"] == drives.PROCESSED / "2026_10_22_10_21AM"
    assert d["camera"]["hood"] == pytest.approx(0.45)
    with pytest.raises(KeyError):
        drives.load("no_such_drive")


def test_street_names_match_regardless_of_diacritics():
    from src.video.truth import norm
    assert norm("Strada Mirceşti") == norm("Strada Mircești") == "mircesti"
    assert norm("Strada Henri Coandă") == "henri coanda"
    assert norm("Strada Columna") != norm("Strada Columna lui Traian")


def test_number_plates_are_dropped_but_street_text_kept():
    from src.video.signs import is_plate
    for plate in ("JDZ• 546", "5NRB 449", "HEMH 369", "C AB 123"):
        assert is_plate(plate), plate
    for text in ("PETRICANI 86", "rompetrol", "Felicia", "RESTAURANT ITALIAN", "Sens unic"):
        assert not is_plate(text), text


def test_house_number_plate_matches_street():
    from src.video.signs import match
    assert match("PETRICANI 86", {"petricani": "Strada Petricani"})[0] == "Strada Petricani"


def test_lane_counting_from_a_scanline():
    from src.video.lanes import frame_lanes
    # curb | road | marking | road (us, centre 480) | marking | road | curb   on a 960 px row
    runs = [["curb", 0, 40], ["road", 41, 300], ["lane_marking", 301, 310], ["road", 311, 640],
            ["lane_marking", 641, 650], ["road", 651, 900], ["curb", 901, 959]]
    r = frame_lanes(runs, 960)
    assert r["lanes_visible"] == 3 and r["lane_from_right"] == 2


def test_event_alignment_prefers_the_matching_route():
    import pandas as pd
    from src.video.event_match import align
    ev = pd.DataFrame(dict(t=[10.0, 50.0], turn_deg=[90, 0], roundabout=[False, False],
                           crosswalk=[False, True], signal=[False, False], cues=["turn", "crosswalk"]))
    def sites(turn):
        s = pd.DataFrame(dict(d=[100.0, 500.0], node=[1, 2], turn=[turn, 0.0], roundabout=[False, False],
                              junction=[True, True], signal=[False, False], crossing=[False, True]))
        s.attrs["length"] = 600.0
        return s
    m_ev, total = [100.0, 500.0], 600.0
    right, _ = align(ev, sites(+88.0), m_ev, total)
    left, _ = align(ev, sites(-88.0), m_ev, total)
    assert right < left                      # the route with the right turn explains the video better


def test_overlay_clean_drops_repeated_misreads():
    import pandas as pd
    from datetime import datetime, timedelta
    from src.video.overlay import clean
    t = list(range(60))
    lat = [47.05 + 0.00005 * i for i in t]                     # ~5.5 m/s straight north
    for i in range(30, 34):                                     # 0↔6 misread repeated 4 s: ~1 km off
        lat[i] += 0.0099
    df = pd.DataFrame(dict(video_s=t, lat=lat, lon=[28.8] * 60,
                           clock=[datetime(2026, 10, 23, 10) + timedelta(seconds=i) for i in t]))
    out = clean(df)
    assert out.bad[30:34].all() and not out.bad[:30].any() and not out.bad[34:].any()


def test_event_alignment_survives_unknown_way_out_of_the_start():
    import numpy as np
    import pandas as pd
    from src.video.event_match import align
    # the car drove 800 m (a yard, a U-turn) before reaching the route: odometer is +800 m throughout
    ev = pd.DataFrame(dict(t=[10.0, 40.0, 70.0], turn_deg=[90, 0, -90], roundabout=[False] * 3,
                           crosswalk=[False, True, False], signal=[False] * 3, cues=["turn", "crosswalk", "turn"]))
    s = pd.DataFrame(dict(d=[300.0, 1000.0, 1700.0], node=[1, 2, 3], turn=[88.0, 0.0, -85.0], roundabout=[False] * 3,
                          junction=[True] * 3, signal=[False] * 3, crossing=[False, True, False]))
    s.attrs["length"] = 2000.0
    m_ev = np.array([1100.0, 1800.0, 2500.0])                   # odometer; total 2800 (800 m extra)
    _, pairs = align(ev, s, m_ev, 2800.0)
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_lens_calibration_from_map_turns():
    import numpy as np
    from src.video.calibrate import fit
    map_turns = np.array([90.0, -85, 60, -120, 45, 95, -40])
    good = fit(np.column_stack([map_turns, 1.5 * map_turns]), 120)   # video turns 1.5× too large
    assert good["status"] == "calibrated"
    assert abs(good["hfov_deg"] - 2 * np.degrees(np.arctan(np.tan(np.radians(60)) / 1.5))) < 0.5   # ~98°
    noisy = fit(np.column_stack([map_turns, [80, 70, -50, -100, 40, -90, 30]]), 120)   # wrong route: no agreement
    assert noisy["status"] == "kept nominal" and noisy["hfov_deg"] == 120
    few = fit(np.column_stack([map_turns[:3], map_turns[:3]]), 120)
    assert few["status"] == "kept nominal"


def test_ground_scale_from_parked_cars():
    import numpy as np
    from src.video.ground import fit
    rng = np.random.default_rng(0)
    yh, S = 270.0, 900.0                                   # true horizon row, focal × camera height
    cars = []
    for z0 in rng.uniform(18, 30, 15):                     # parked cars 18–30 m ahead
        s = np.linspace(0, 10, 8)                          # we drive 10 m past each
        cars.append((yh + S / (z0 - s) + rng.normal(0, 0.5, len(s)), s))
    r = fit(cars, y_guess=280.0)
    assert r["ground_status"] == "calibrated"
    assert abs(r["horizon_px"] - yh) <= 2 and abs(r["ground_scale"] - S) / S < 0.1


def test_ego_motion_moves_points_between_car_frames():
    import numpy as np
    from src.video.surround import to_earlier
    # straight: a point 6 m ahead after driving 4 m was 10 m ahead before
    X, Z = to_earlier(0.0, 6.0, d=4.0, a=0.0)
    assert abs(X) < 1e-9 and abs(Z - 10.0) < 1e-9
    # turned 90° right on the spot: 10 m ahead now = 10 m to the right before
    X, Z = to_earlier(0.0, 10.0, d=0.0, a=np.pi / 2)
    assert abs(X - 10.0) < 1e-9 and abs(Z) < 1e-9


def test_speed_from_flow_and_depth_ignores_turning_and_windscreen_dirt():
    import numpy as np
    from src.video.speed import fit_motion
    rng = np.random.default_rng(1)
    n = 3000
    x, y = rng.uniform(-0.8, 0.8, n), rng.uniform(-0.4, 0.3, n)     # focal-length units
    Z = rng.uniform(4, 80, n)
    ey, T, wx, wy, wz = 0.05, 0.5, 0.001, 0.01, 0.0005              # 0.5 m per frame, turning right
    du = (x - 0.0) * T / Z + x * y * wx - (1 + x ** 2) * wy + y * wz
    dv = (y - ey) * T / Z + (1 + y ** 2) * wx - x * y * wy - x * wz
    du, dv = du + rng.normal(0, 0.0005, n), dv + rng.normal(0, 0.0005, n)
    dirt = rng.random(n) < 0.25                                      # raindrops: tracked but still
    du[dirt], dv[dirt] = 0.0, 0.0
    t, rot, _, _ = fit_motion(x, y, du, dv, Z, 0.0, ey, still_n=0.3 / 600)
    assert abs(t - T) / T < 0.08
    assert abs(rot[1] - wy) < 0.002
    # standing still with only dirt and noise: no speed
    t0, _, _, _ = fit_motion(x, y, rng.normal(0, 1e-4, n), rng.normal(0, 1e-4, n), Z, 0.0, ey, still_n=0.3 / 600)
    assert abs(t0) < 0.05
