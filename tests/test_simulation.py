"""The simulator must be reproducible and must never derive bus delays from the parser."""
from __future__ import annotations

import ast
import inspect
import sqlite3

import pandas as pd

import src.ingest
import src.simulate
from src.ingest import load_simulation
from src.nlp_parser import ParsedBulletin
from src.simulate import SimConfig, simulate

CFG = SimConfig(days=5, seed=7)


def test_same_config_gives_identical_data():
    a, b = simulate(CFG), simulate(CFG)
    pd.testing.assert_frame_equal(a.bus, b.bus)
    pd.testing.assert_frame_equal(a.rail, b.rail)


def test_placebo_changes_only_the_spillover():
    planted = simulate(CFG)
    placebo = simulate(SimConfig(days=CFG.days, seed=CFG.seed, planted_effect=False))
    pd.testing.assert_frame_equal(planted.rail, placebo.rail)  # same incidents, same bulletins
    assert (placebo.bus["truth_spillover_sec"] == 0).all()
    assert (planted.bus["truth_spillover_sec"] > 0).any()
    no_spill = planted.bus["truth_spillover_sec"] == 0
    pd.testing.assert_series_equal(planted.bus.loc[no_spill, "mean_delay_sec"],
                                   placebo.bus.loc[no_spill, "mean_delay_sec"])


def test_simulator_does_not_import_the_parser():
    tree = ast.parse(inspect.getsource(src.simulate))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("nlp_parser" in (m or "") for m in imported)


def test_bus_delays_do_not_depend_on_the_parser(monkeypatch, tmp_path):
    """Swap in a nonsense parser: bulletin readings change, bus delays must not."""
    nonsense = ParsedBulletin(1, "x", "x", 1, 0, 1, 1.0)
    monkeypatch.setattr(src.ingest, "parse_bulletin", lambda text, status: nonsense)
    db = tmp_path / "sim.db"
    load_simulation(CFG, db_path=db)
    with sqlite3.connect(db) as conn:
        bus = pd.read_sql("SELECT * FROM bus_telemetry ORDER BY route_id, bucket_15m_epoch", conn)
        severities = pd.read_sql("SELECT DISTINCT severity_score FROM rail_alerts", conn)["severity_score"]
    expected = simulate(CFG).bus.drop(columns="truth_spillover_sec").sort_values(["route_id", "bucket_15m_epoch"])
    pd.testing.assert_frame_equal(bus, expected.reset_index(drop=True), check_dtype=False)
    assert list(severities) == [1.0]  # the nonsense parser really was used


def test_bulletins_exercise_the_tricky_cases():
    texts = " ".join(simulate(SimConfig(days=28, seed=42)).rail["text_ja"])
    for phrase in ("運転を再開しました", "振替輸送は行っておりません", "15分以上の遅延はありません", "平常どおり運転"):
        assert phrase in texts
