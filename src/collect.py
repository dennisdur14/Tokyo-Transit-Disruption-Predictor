"""Collect one snapshot of live Tokyo data and save it as a small gzipped JSON file.

    python -m src.collect                         # -> data/raw/YYYY/MM/DD/snapshot_<JST time>.json.gz
    python -m src.collect --out datastore/raw     # what the GitHub Action runs every 15 minutes

Each snapshot records:
  * every Toei bus position (keyless GTFS-Realtime feed), with an estimated delay per bus
    worked out against Toei's published GTFS timetable;
  * rail status bulletins from ODPT: Toei lines without a key, plus JR East, Tokyo Metro,
    Tokyu, ... when ODPT_API_KEY is set;
  * current Tokyo weather from Open-Meteo.

A source that fails is noted under "errors" and the rest is still saved. The run exits
with an error only if nothing at all could be collected.

Only the standard library, requests and the GTFS-Realtime bindings are used, so the
scheduled job installs in seconds.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import statistics
import sys
import unicodedata
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import (
    DATA_DIR,
    ODPT_API_KEY,
    ODPT_PUBLIC_TRAIN_INFO_URL,
    ODPT_TRAIN_INFO_URL,
    OPEN_METEO_URL,
    RAW_DIR,
    TOEI_BUS_GTFS_URL,
    TOEI_BUS_POSITIONS_URL,
    TOKYO_LAT,
    TOKYO_LON,
)

log = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
STALE_AFTER_SEC = 300  # ignore bus positions older than 5 minutes
TIMEOUT = 30


def nfkc(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


# ----------------------------------------------------------------------------------------
# Toei bus timetable (static GTFS)
# ----------------------------------------------------------------------------------------
def timetable_zip(cache_dir: Path, today: str) -> Path:
    """Path to today's timetable zip, downloading it if needed (falls back to an older copy)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"ToeiBus-GTFS-{today}.zip"
    if path.exists():
        return path
    older = sorted(cache_dir.glob("ToeiBus-GTFS-*.zip"))
    try:
        resp = requests.get(TOEI_BUS_GTFS_URL, timeout=120)
        resp.raise_for_status()
        zipfile.ZipFile(io.BytesIO(resp.content)).testzip()  # fail early on a broken download
        path.write_bytes(resp.content)
        for old in older:
            old.unlink()
        return path
    except Exception:
        if older:
            log.warning("Timetable download failed; using older copy %s", older[-1].name)
            return older[-1]
        raise


def _seconds(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)  # GTFS hours can exceed 24 for after-midnight trips


def load_timetable(zip_path: Path, trip_ids: set[str]) -> tuple[dict[str, str], dict[str, list[tuple[int, int]]]]:
    """Return (route_id -> short name, trip_id -> sorted [(stop_sequence, scheduled seconds)])."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open("routes.txt") as fh:
            routes = {r["route_id"]: nfkc(r.get("route_short_name")) for r in
                      csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig"))}
        schedule: dict[str, list[tuple[int, int]]] = {}
        with z.open("stop_times.txt") as fh:
            reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
            header = next(reader)
            i_trip, i_seq, i_arr = (header.index(c) for c in ("trip_id", "stop_sequence", "arrival_time"))
            for row in reader:  # ~1.3M rows; keep only trips that are running right now
                if row[i_trip] in trip_ids and row[i_arr]:
                    schedule.setdefault(row[i_trip], []).append((int(row[i_seq]), _seconds(row[i_arr])))
    for stops in schedule.values():
        stops.sort()
    return routes, schedule


def estimate_delay(vehicle_epoch: int, stop_sequence: int, stops: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Estimated delay in seconds (positive = late), plus the scheduled gap it sits in.

    The feed says which stop a bus is heading to (stop k), so it has left stop k-1 but not
    reached stop k. Its delay is taken against the midpoint of the two scheduled times,
    which is accurate to about half the gap between stops (typically ~1 minute).
    """
    seqs = [s for s, _ in stops]
    if stop_sequence not in seqs:
        return None
    i = seqs.index(stop_sequence)
    if i == 0:
        return None  # still at the first stop, so it hasn't departed yet
    prev_s, next_s = stops[i - 1][1], stops[i][1]
    # Timetable seconds count from the service day's midnight; a trip running after
    # midnight belongs to yesterday's service day. Use whichever day fits best.
    local = datetime.fromtimestamp(vehicle_epoch, JST)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = [vehicle_epoch - (int((midnight - timedelta(days=d)).timestamp()) + (prev_s + next_s) / 2)
                  for d in (0, 1)]
    delay = min(candidates, key=abs)
    return int(round(delay)), next_s - prev_s


# ----------------------------------------------------------------------------------------
# Live sources
# ----------------------------------------------------------------------------------------
def collect_buses(cache_dir: Path, today: str) -> tuple[list[dict], dict]:
    from google.transit import gtfs_realtime_pb2

    resp = requests.get(TOEI_BUS_POSITIONS_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    vehicles = [e.vehicle for e in feed.entity if e.HasField("vehicle")]
    info = {"bus_feed_epoch": int(feed.header.timestamp)}

    routes, schedule = {}, {}
    try:
        zip_path = timetable_zip(cache_dir, today)
        routes, schedule = load_timetable(zip_path, {v.trip.trip_id for v in vehicles})
        info["timetable"] = zip_path.name
    except Exception as exc:  # positions are still worth keeping without delays
        info["timetable_error"] = f"{type(exc).__name__}: {exc}"

    rows = []
    for v in vehicles:
        est = None
        if v.timestamp and feed.header.timestamp - v.timestamp <= STALE_AFTER_SEC and v.trip.trip_id in schedule:
            est = estimate_delay(int(v.timestamp), int(v.current_stop_sequence), schedule[v.trip.trip_id])
        rows.append({
            "vehicle_id": v.vehicle.id,
            "trip_id": v.trip.trip_id,
            "route_id": v.trip.route_id,
            "route_short_name": routes.get(v.trip.route_id, ""),
            "direction_id": v.trip.direction_id if v.trip.HasField("direction_id") else None,
            "stop_sequence": int(v.current_stop_sequence),
            "stop_id": v.stop_id,
            "vehicle_epoch": int(v.timestamp),
            "lat": round(v.position.latitude, 5),
            "lon": round(v.position.longitude, 5),
            "delay_s": est[0] if est else None,
            "sched_gap_s": est[1] if est else None,
        })
    return rows, info


def _train_info_record(item: dict, source: str) -> dict:
    status = item.get("odpt:trainInformationStatus") or {}
    text = item.get("odpt:trainInformationText") or {}
    pick = lambda obj, lang: obj.get(lang, "") if isinstance(obj, dict) else str(obj)  # noqa: E731
    return {
        "railway": item.get("odpt:railway"),  # None for operator-wide notices
        "operator": item.get("odpt:operator"),
        "status_ja": pick(status, "ja"),
        "status_en": pick(status, "en"),
        "text_ja": pick(text, "ja"),
        "text_en": pick(text, "en"),
        "dc_date": item.get("dc:date"),
        "time_of_origin": item.get("odpt:timeOfOrigin"),
        "source": source,
    }


def collect_rail(api_key: str) -> tuple[list[dict], dict]:
    errors = {}
    records: dict[tuple, dict] = {}
    sources = [("public", ODPT_PUBLIC_TRAIN_INFO_URL, {})]
    if api_key:
        sources.append(("keyed", ODPT_TRAIN_INFO_URL, {"acl:consumerKey": api_key}))
    else:
        errors["rail_keyed"] = "skipped: no ODPT_API_KEY (only keyless Toei lines collected)"
    for name, url, params in sources:
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            for item in resp.json():
                rec = _train_info_record(item, name)
                records[(rec["railway"], rec["operator"])] = rec  # keyed copy wins if both
        except Exception as exc:
            errors[f"rail_{name}"] = f"{type(exc).__name__}: {exc}"
    return list(records.values()), errors


def collect_weather() -> dict:
    resp = requests.get(OPEN_METEO_URL, timeout=TIMEOUT, params={
        "latitude": TOKYO_LAT, "longitude": TOKYO_LON,
        "current": "temperature_2m,precipitation,wind_speed_10m,weather_code", "wind_speed_unit": "ms",
    })
    resp.raise_for_status()
    cur = resp.json()["current"]
    return {"time_utc": cur["time"], "temp_c": cur["temperature_2m"], "precip_mm_15min": cur["precipitation"],
            "wind_speed_ms": cur["wind_speed_10m"], "weather_code": cur["weather_code"]}


# ----------------------------------------------------------------------------------------
def collect_snapshot(out_dir: Path = RAW_DIR, cache_dir: Path = DATA_DIR / "gtfs_cache",
                     api_key: str = ODPT_API_KEY) -> Path | None:
    now = datetime.now(JST)
    snap: dict = {"schema": 1, "collected_at": now.isoformat(timespec="seconds"),
                  "collected_epoch": int(now.timestamp()), "errors": {}, "buses": [], "rail": [], "weather": None}
    try:
        snap["buses"], info = collect_buses(cache_dir, now.strftime("%Y%m%d"))
        snap.update(info)
        if "timetable_error" in info:
            snap["errors"]["timetable"] = info.pop("timetable_error")
    except Exception as exc:
        snap["errors"]["buses"] = f"{type(exc).__name__}: {exc}"
    snap["rail"], rail_errors = collect_rail(api_key)
    snap["errors"].update(rail_errors)
    try:
        snap["weather"] = collect_weather()
    except Exception as exc:
        snap["errors"]["weather"] = f"{type(exc).__name__}: {exc}"

    delays = [b["delay_s"] for b in snap["buses"] if b["delay_s"] is not None]
    log.info("buses: %d (%d with delay estimate%s) | rail bulletins: %d | weather: %s | errors: %s",
             len(snap["buses"]), len(delays), f", median {statistics.median(delays) / 60:.1f} min" if delays else "",
             len(snap["rail"]), "ok" if snap["weather"] else "missing", snap["errors"] or "none")
    if not snap["buses"] and not snap["rail"] and not snap["weather"]:
        return None
    path = Path(out_dir) / now.strftime("%Y/%m/%d") / f"snapshot_{now.strftime('%Y%m%dT%H%M%S')}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(snap, fh, ensure_ascii=False, separators=(",", ":"))
    log.info("saved %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Save one snapshot of live Tokyo transit data.")
    parser.add_argument("--out", default=str(RAW_DIR), help="folder for snapshot files")
    parser.add_argument("--gtfs-cache", default=str(DATA_DIR / "gtfs_cache"), help="folder for the timetable zip")
    args = parser.parse_args()
    if collect_snapshot(Path(args.out), Path(args.gtfs_cache)) is None:
        sys.exit("Nothing could be collected (all sources failed).")


if __name__ == "__main__":
    main()
