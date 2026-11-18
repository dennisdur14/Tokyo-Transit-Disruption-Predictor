"""Load data into the SQLite database -- either a simulation or real collected snapshots.

    python -m src.ingest simulate                 # synthetic data with a planted effect
    python -m src.ingest simulate --placebo       # same world, effect switched off
    python -m src.ingest snapshots                # real data saved by src.collect

Both paths fill the same tables, so everything downstream works the same way on either.
The bulletin parser runs here, at load time, exactly as it would on live data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BUCKET_SECONDS,
    BUS_ROUTES,
    RAIL_LINES,
    RAW_DIR,
    ROUTE_RAIL_LINKS,
    SQL_DIR,
    get_connection,
)
from src.nlp_parser import normalize, parse_bulletin, status_category
from src.simulate import JST, SimConfig, Simulation, simulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FACT_TABLES = ("rail_alerts", "bus_telemetry", "weather_hourly", "metadata")


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and (re)write the reference data from src/config.py."""
    conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executemany(
        "INSERT OR REPLACE INTO rail_lines VALUES (?, ?, ?, ?, ?)",
        [(l.line_id, l.operator, l.name_en, l.name_ja, int(l.needs_key)) for l in RAIL_LINES],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO bus_routes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r.route_id, r.route_code, r.gtfs_short_name, r.name_en, r.name_ja, r.lat, r.lon) for r in BUS_ROUTES],
    )
    conn.execute("DELETE FROM route_rail_links")
    conn.executemany("INSERT INTO route_rail_links VALUES (?, ?, ?)", ROUTE_RAIL_LINKS)
    conn.commit()


def _reset_facts(conn: sqlite3.Connection) -> None:
    for table in FACT_TABLES:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def parse_rail_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add the header category and the parser's fields to raw (line, time, header, text) rows."""
    parsed = [parse_bulletin(t, s) for t, s in zip(df["text_ja"], df["status_ja"])]
    out = df[["line_id", "observed_epoch", "status_ja", "text_ja"]].copy()
    out.insert(0, "alert_id", [
        hashlib.sha1(f"{l}:{e}".encode()).hexdigest()[:16] for l, e in zip(df["line_id"], df["observed_epoch"])
    ])
    out["status_category"] = [status_category(s) for s in df["status_ja"]]
    out["is_disrupted"] = [p.is_disrupted for p in parsed]
    out["parsed_cause"] = [p.parsed_cause for p in parsed]
    out["is_suspended"] = [p.is_suspended for p in parsed]
    out["is_resumed"] = [p.is_resumed for p in parsed]
    out["has_furikae_yuso"] = [p.has_furikae_yuso for p in parsed]
    out["severity_score"] = [p.severity_score for p in parsed]
    return out


def _write_metadata(conn: sqlite3.Connection, items: dict[str, object]) -> None:
    items = {**items, "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    conn.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", [(k, str(v)) for k, v in items.items()])
    conn.commit()


def read_metadata(db_path: str | Path | None = None) -> dict[str, str]:
    conn = get_connection(db_path)
    try:
        return dict(conn.execute("SELECT key, value FROM metadata").fetchall())
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


# ----------------------------------------------------------------------------------------
# Simulated data
# ----------------------------------------------------------------------------------------
def load_simulation(cfg: SimConfig = SimConfig(), db_path: str | Path | None = None) -> Simulation:
    sim = simulate(cfg)
    conn = get_connection(db_path)
    try:
        init_db(conn)
        _reset_facts(conn)
        parse_rail_rows(sim.rail).to_sql("rail_alerts", conn, if_exists="append", index=False)
        sim.bus.drop(columns=["truth_spillover_sec"]).to_sql("bus_telemetry", conn, if_exists="append", index=False)
        sim.weather.to_sql("weather_hourly", conn, if_exists="append", index=False)
        _write_metadata(conn, {
            "data_source": "simulated",
            "sim_start": cfg.start.isoformat(),
            "sim_days": cfg.days,
            "sim_seed": cfg.seed,
            "sim_planted_effect": cfg.planted_effect,
        })
        log.info("Simulated %d days (seed %d, planted effect %s): %d rail rows, %d bus rows, %d incidents.",
                 cfg.days, cfg.seed, cfg.planted_effect, len(sim.rail), len(sim.bus), len(sim.incidents))
    finally:
        conn.close()
    return sim


# ----------------------------------------------------------------------------------------
# Real collected snapshots (written by src.collect)
# ----------------------------------------------------------------------------------------
def read_snapshots(raw_dir: str | Path = RAW_DIR):
    """Yield each snapshot dict, oldest first."""
    for path in sorted(Path(raw_dir).rglob("snapshot_*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                yield json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping unreadable snapshot %s: %s", path, exc)


def load_snapshots(raw_dir: str | Path = RAW_DIR, db_path: str | Path | None = None) -> dict[str, int]:
    rail_rows, vehicle_rows, weather_rows, n_snapshots = [], [], [], 0
    for snap in read_snapshots(raw_dir):
        n_snapshots += 1
        t = int(snap["collected_epoch"])
        for r in snap.get("rail", []):
            rail_rows.append({"line_id": r["railway"], "observed_epoch": t,
                              "status_ja": r.get("status_ja") or "", "text_ja": r.get("text_ja") or ""})
        for v in snap.get("buses", []):
            if v.get("delay_s") is not None:
                vehicle_rows.append({"short_name": normalize(v.get("route_short_name")),
                                     "trip_id": v["trip_id"], "vehicle_epoch": int(v["vehicle_epoch"]),
                                     "delay_s": float(v["delay_s"])})
        if snap.get("weather"):
            w = snap["weather"]
            weather_rows.append({"epoch": t, "temp_c": w["temp_c"], "precip_15m": w["precip_mm_15min"],
                                 "wind_speed_ms": w["wind_speed_ms"], "weather_code": w["weather_code"]})
    if n_snapshots == 0:
        raise SystemExit(f"No snapshot_*.json.gz files under {raw_dir}. Run `python -m src.collect` first.")

    conn = get_connection(db_path)
    try:
        init_db(conn)
        _reset_facts(conn)
        rail = pd.DataFrame(rail_rows).drop_duplicates(["line_id", "observed_epoch"])
        _add_unknown_lines(conn, rail["line_id"].unique())
        parse_rail_rows(rail).to_sql("rail_alerts", conn, if_exists="append", index=False)

        bus = _aggregate_bus(pd.DataFrame(vehicle_rows))
        bus.to_sql("bus_telemetry", conn, if_exists="append", index=False)

        weather = _aggregate_weather(pd.DataFrame(weather_rows))
        weather.to_sql("weather_hourly", conn, if_exists="append", index=False)

        seen = set(rail["line_id"])
        for line_id in sorted({l for _, l, _ in ROUTE_RAIL_LINKS} - seen):
            log.warning("Linked line %s never appears in the snapshots -- missing ODPT key, or wrong railway ID?", line_id)
        first, last = int(rail["observed_epoch"].min()), int(rail["observed_epoch"].max())
        _write_metadata(conn, {
            "data_source": "collected",
            "n_snapshots": n_snapshots,
            "first_snapshot": datetime.fromtimestamp(first, JST).isoformat(),
            "last_snapshot": datetime.fromtimestamp(last, JST).isoformat(),
        })
        counts = {"snapshots": n_snapshots, "rail_rows": len(rail), "bus_rows": len(bus), "weather_rows": len(weather)}
        log.info("Loaded collected data: %s", counts)
        return counts
    finally:
        conn.close()


def _add_unknown_lines(conn: sqlite3.Connection, line_ids) -> None:
    """Keep every line the feed reports, even ones not in config (e.g. Toei Asakusa)."""
    known = {row[0] for row in conn.execute("SELECT line_id FROM rail_lines")}
    for line_id in sorted(set(line_ids) - known):
        name = line_id.split(":")[-1]
        conn.execute("INSERT INTO rail_lines VALUES (?, ?, ?, ?, 1)", (line_id, name.split(".")[0], name, name))
    conn.commit()


def _aggregate_bus(vehicles: pd.DataFrame) -> pd.DataFrame:
    cols = ["route_id", "bucket_15m_epoch", "n_vehicles", "mean_delay_sec", "p90_delay_sec"]
    if vehicles.empty:
        return pd.DataFrame(columns=cols)
    route_by_short_name = {r.gtfs_short_name: r.route_id for r in BUS_ROUTES}
    v = vehicles.assign(route_id=vehicles["short_name"].map(route_by_short_name)).dropna(subset=["route_id"])
    v = v.drop_duplicates(["trip_id", "vehicle_epoch"])  # same position report seen twice
    v["bucket_15m_epoch"] = v["vehicle_epoch"] - v["vehicle_epoch"] % BUCKET_SECONDS
    agg = v.groupby(["route_id", "bucket_15m_epoch"])["delay_s"].agg(
        n_vehicles="count", mean_delay_sec="mean", p90_delay_sec=lambda s: float(np.percentile(s, 90))
    ).reset_index()
    return agg[cols].round({"mean_delay_sec": 1, "p90_delay_sec": 1})


def _aggregate_weather(w: pd.DataFrame) -> pd.DataFrame:
    cols = ["hour_epoch", "temp_c", "precip_mm", "wind_speed_ms", "weather_code"]
    if w.empty:
        return pd.DataFrame(columns=cols)
    w = w.assign(hour_epoch=w["epoch"] - w["epoch"] % 3600)
    agg = w.groupby("hour_epoch").agg(
        temp_c=("temp_c", "mean"),
        precip_mm=("precip_15m", lambda s: float(s.mean() * 4)),  # 15-minute amounts -> mm per hour
        wind_speed_ms=("wind_speed_ms", "mean"),
        weather_code=("weather_code", "max"),
    ).reset_index()
    return agg[cols].round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="generate synthetic data with a planted effect")
    s.add_argument("--days", type=int, default=SimConfig.days)
    s.add_argument("--seed", type=int, default=SimConfig.seed)
    s.add_argument("--start", default=SimConfig.start.date().isoformat(), help="first day, YYYY-MM-DD (JST)")
    s.add_argument("--placebo", action="store_true", help="switch the planted rail->bus effect off")
    c = sub.add_parser("snapshots", help="load real snapshots saved by src.collect")
    c.add_argument("--raw-dir", default=str(RAW_DIR))
    args = parser.parse_args()

    if args.command == "simulate":
        start = datetime.fromisoformat(args.start).replace(tzinfo=JST)
        load_simulation(SimConfig(start=start, days=args.days, seed=args.seed, planted_effect=not args.placebo))
    else:
        load_snapshots(args.raw_dir)


if __name__ == "__main__":
    main()
