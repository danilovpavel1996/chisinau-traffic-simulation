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
