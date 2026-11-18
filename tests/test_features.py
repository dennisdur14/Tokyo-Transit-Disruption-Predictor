"""sql/features.sql must only use information available when the forecast is made."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.ingest import load_simulation
from src.simulate import SimConfig
from src.train import load_feature_table

FEATURES = ["n_vehicles", "mean_delay_sec", "p90_delay_sec", "rolling_60m_delay_sec", "lag_15m_delay_sec",
            "lag_30m_delay_sec", "delay_change_15m_sec", "temp_c", "precip_mm", "wind_speed_ms",
            "status_any_abnormal", "status_any_suspended", "status_any_delay", "status_any_resumed",
            "status_strong_abnormal", "rail_any_disrupted", "rail_any_suspended", "rail_any_resumed",
            "rail_any_furikae", "rail_max_severity", "rail_strong_max_severity", "rail_strong_furikae",
            "rail_disrupted_snapshots", "top_rail_cause", "recent_bulletins_ja"]


@pytest.fixture()
def sim_db(tmp_path):
    db = tmp_path / "sim.db"
    load_simulation(SimConfig(days=4, seed=3), db_path=db)
    return db


def test_features_ignore_the_future(sim_db):
    """Scramble everything observed after time T; features known before T must not change."""
    before = load_feature_table(sim_db)
    T = int(np.median(before["bucket_15m_epoch"])) // 3600 * 3600 + 1800  # mid-hour, mid-dataset
    with sqlite3.connect(sim_db) as conn:
        conn.execute("UPDATE bus_telemetry SET mean_delay_sec = mean_delay_sec + 9999, "
                     "p90_delay_sec = p90_delay_sec + 9999, n_vehicles = 99 WHERE bucket_15m_epoch >= ?", (T,))
        conn.execute("UPDATE rail_alerts SET is_disrupted = 1, is_suspended = 1, has_furikae_yuso = 1, "
                     "severity_score = 1.0, status_category = 'suspended', parsed_cause = 'track_injury_accident', "
                     "text_ja = 'FUTURE' WHERE observed_epoch >= ?", (T,))
        # an hour's weather is only complete at the end of that hour
        conn.execute("UPDATE weather_hourly SET precip_mm = 999, temp_c = 99, wind_speed_ms = 99 "
                     "WHERE hour_epoch + 3600 > ?", (T,))
    after = load_feature_table(sim_db)

    known = lambda df: df[df["bucket_15m_epoch"] + 900 <= T].set_index(["route_id", "bucket_15m_epoch"])  # noqa: E731
    pd.testing.assert_frame_equal(known(before)[FEATURES], known(after)[FEATURES])
    # ...while the scrambling did reach rows after T (so the test is not vacuous)
    assert not after.loc[after["bucket_15m_epoch"] >= T, "mean_delay_sec"].equals(
        before.loc[before["bucket_15m_epoch"] >= T, "mean_delay_sec"])


def test_missing_bucket_never_shifts_lags_or_targets(sim_db):
    df = load_feature_table(sim_db)
    route = "Toei.To01"
    present = set(df.loc[df["route_id"] == route, "bucket_15m_epoch"])
    # a bucket whose neighbours all exist, so any NULL below is caused by the deletion
    gap = next(int(b) for b in sorted(present)[len(present) // 2:]
               if {b - 1800, b - 900, b + 900, b + 1800} <= present)
    with sqlite3.connect(sim_db) as conn:
        conn.execute("DELETE FROM bus_telemetry WHERE route_id = ? AND bucket_15m_epoch = ?", (route, gap))
    out = load_feature_table(sim_db).set_index(["route_id", "bucket_15m_epoch"])
    assert pd.isna(out.loc[(route, gap - 1800), "target_delay_30m_sec"])  # its "30 min later" is gone
    assert pd.isna(out.loc[(route, gap + 900), "lag_15m_delay_sec"])  # its "15 min earlier" is gone
    assert pd.isna(out.loc[(route, gap + 1800), "lag_30m_delay_sec"])


def test_one_row_per_route_and_bucket(sim_db):
    df = load_feature_table(sim_db)
    assert not df.duplicated(["route_id", "bucket_15m_epoch"]).any()
