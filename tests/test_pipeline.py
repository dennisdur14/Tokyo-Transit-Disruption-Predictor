"""End-to-end runs: simulation -> report, and collected snapshots -> database -> labels."""
from __future__ import annotations

import gzip
import json
import math
import sqlite3

import pandas as pd

from src.ingest import load_simulation, load_snapshots
from src.parser_eval import export, score
from src.simulate import SimConfig
from src.train import MODEL_NAMES, run


def test_simulation_to_report(tmp_path):
    db, art = tmp_path / "sim.db", tmp_path / "artifacts"
    load_simulation(SimConfig(days=10, seed=11), db_path=db)
    report = run(db_path=db, artifacts_dir=art)
    assert [m["model"] for m in report["models"]] == MODEL_NAMES
    for m in report["models"]:
        assert math.isfinite(m["rmse_sec"]) and len(m["rmse_by_test_block"]) == 4
    scores = {m["model"]: m["rmse_sec"] for m in report["models"]}
    assert scores[MODEL_NAMES[2]] < scores[MODEL_NAMES[1]]  # recent delays beat time-of-day alone
    for name in ("ablation_report.json", "metrics_summary.csv", "model_bundle.joblib", "ablation_comparison.png"):
        assert (art / name).exists()


def _write_snapshot(raw_dir, epoch: int, delay: int, text: str) -> None:
    snap = {
        "schema": 1, "collected_at": "", "collected_epoch": epoch, "errors": {},
        "buses": [{"vehicle_id": "V1", "trip_id": f"T{epoch}", "route_id": "006", "route_short_name": "都01(T01)",
                   "direction_id": 1, "stop_sequence": 4, "stop_id": "x", "vehicle_epoch": epoch - 30,
                   "lat": 35.66, "lon": 139.73, "delay_s": delay, "sched_gap_s": 120}],
        "rail": [{"railway": "odpt.Railway:Toei.Oedo", "operator": "odpt.Operator:Toei", "status_ja": "",
                  "status_en": "", "text_ja": text, "text_en": "", "dc_date": "", "time_of_origin": "",
                  "source": "public"}],
        "weather": {"time_utc": "", "temp_c": 15.0, "precip_mm_15min": 0.5, "wind_speed_ms": 2.0, "weather_code": 61},
    }
    path = raw_dir / f"snapshot_{epoch}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False)


def test_snapshots_to_database_and_labels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    normal, suspended = "現在、１５分以上の遅延はありません。", "人身事故の影響で、全線で運転を見合わせています。"
    t0 = 1_794_960_000  # a 15-minute boundary
    _write_snapshot(raw, t0 + 60, 120, normal)
    _write_snapshot(raw, t0 + 960, 300, suspended)
    db = tmp_path / "real.db"
    counts = load_snapshots(raw, db_path=db)
    assert counts == {"snapshots": 2, "rail_rows": 2, "bus_rows": 2, "weather_rows": 1}
    with sqlite3.connect(db) as conn:
        rail = pd.read_sql("SELECT text_ja, is_disrupted, is_suspended FROM rail_alerts ORDER BY observed_epoch", conn)
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        weather = conn.execute("SELECT precip_mm FROM weather_hourly").fetchone()[0]
    assert rail["is_disrupted"].tolist() == [0, 1] and rail["is_suspended"].tolist() == [0, 1]
    assert meta["data_source"] == "collected"
    assert weather == 2.0  # 0.5 mm per 15 minutes -> 2 mm per hour

    labels = tmp_path / "labels.csv"
    assert export(raw, labels) == 2 and export(raw, labels) == 0  # second export adds nothing
    df = pd.read_csv(labels, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.loc[df["text_ja"] == normal, ["label_is_disrupted", "label_is_suspended"]] = ["0", "0"]
    df.loc[df["text_ja"] == suspended, ["label_is_disrupted", "label_is_suspended"]] = ["1", "1"]
    df.to_csv(labels, index=False, encoding="utf-8-sig")
    result = score(labels, artifacts_dir=tmp_path / "art")
    assert result["fields"]["is_disrupted"]["accuracy"] == 1.0
    assert result["fields"]["is_suspended"]["recall"] == 1.0
