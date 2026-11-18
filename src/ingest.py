from __future__ import annotations

import argparse
import hashlib
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import (
    BUS_ROUTES,
    ODPT_API_KEY,
    ODPT_TRAIN_INFO_URL,
    RAIL_LINES,
    SQL_DIR,
    TOEI_TRIP_UPDATE_URL,
    get_connection,
)
from src.nlp_parser import parse_japanese_bulletin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

DISRUPTION_TEMPLATES: list[tuple[str, str]] = [
    (
        "運転見合わせ",
        "【{line_ja} 運転見合わせ】{hub}駅構内での人身事故の影響で、上下線で運転を見合わせています。現在、都営バスおよび周辺路線にて振替輸送を実施しています。",
    ),
    (
        "遅延",
        "【{line_ja} 遅延】{hub}駅での車両点検の影響で、上下線の一部列車に20〜35分程度の遅れと運休が出ています。振替輸送をご利用ください。",
    ),
    (
        "ダイヤ乱れ",
        "【{line_ja} ダイヤ乱れ】強風による速度規制および信号確認の影響で、直通運転を中止しダイヤが大幅に乱れています。",
    ),
    (
        "遅延",
        "【{line_ja} 遅延】{hub}駅での急病人救護および安全確認の影響で、一部列車に15分前後の遅れが出ています。",
    ),
    (
        "運転見合わせ",
        "【{line_ja} 運転見合わせ】信号装置故障の影響で一部区間の運転を見合わせています。都営バスへの振替輸送を実施中です。",
    ),
]


def init_schema(conn) -> None:
    schema_sql = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    cursor = conn.cursor()
    cursor.executescript(schema_sql)

    for rl in RAIL_LINES:
        cursor.execute(
            """
            INSERT INTO rail_lines (line_id, operator_name, line_name_en, line_name_ja, primary_hub, hub_lat, hub_lon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(line_id) DO UPDATE SET
                operator_name=excluded.operator_name,
                line_name_en=excluded.line_name_en,
                line_name_ja=excluded.line_name_ja,
                primary_hub=excluded.primary_hub,
                hub_lat=excluded.hub_lat,
                hub_lon=excluded.hub_lon
            """,
            (rl.line_id, rl.operator_name, rl.line_name_en, rl.line_name_ja, rl.primary_hub, rl.hub_lat, rl.hub_lon),
        )

    for br in BUS_ROUTES:
        cursor.execute(
            """
            INSERT INTO bus_routes (route_id, route_code, route_name_en, route_name_ja, anchor_hub, nearest_rail_line_id, hub_distance_km, baseline_headway_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(route_id) DO UPDATE SET
                route_code=excluded.route_code,
                route_name_en=excluded.route_name_en,
                route_name_ja=excluded.route_name_ja,
                anchor_hub=excluded.anchor_hub,
                nearest_rail_line_id=excluded.nearest_rail_line_id,
                hub_distance_km=excluded.hub_distance_km,
                baseline_headway_sec=excluded.baseline_headway_sec
            """,
            (
                br.route_id,
                br.route_code,
                br.route_name_en,
                br.route_name_ja,
                br.anchor_hub,
                br.nearest_rail_line_id,
                br.hub_distance_km,
                br.baseline_headway_sec,
            ),
        )
    conn.commit()


def _floor_15m(epoch: int) -> int:
    return epoch - (epoch % 900)


def _floor_hour(epoch: int) -> int:
    return epoch - (epoch % 3600)


def generate_simulated_history(db_url: str | None = None, days: int = 14, seed: int = 42) -> None:
    """Populate SQLite/PostgreSQL with 14 days of realistic Tokyo multimodal transit telemetry."""
    rng = np.random.default_rng(seed)
    conn = get_connection(db_url)
    try:
        init_schema(conn)

        end_dt = datetime.now(JST).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=days)
        total_buckets = days * 96

        logger.info("Generating %d days (%d 15-min intervals) of Tokyo multimodal telemetry...", days, total_buckets)

        # 1. Hourly Tokyo weather
        weather_rows: list[dict[str, Any]] = []
        rain_regime = 0.0
        for h in range(days * 24 + 2):
            dt_h = start_dt + timedelta(hours=h)
            epoch_h = int(dt_h.timestamp())
            hour_jst = dt_h.hour

            if rng.random() < 0.06:
                rain_regime = float(rng.uniform(2.5, 14.0))
            else:
                rain_regime = max(0.0, rain_regime * 0.72 - 0.2)

            temp_c = 15.5 + 4.5 * math.sin((hour_jst - 8) * math.pi / 12.0) + float(rng.normal(0, 1.1))
            wind_ms = 2.8 + (3.5 if rain_regime > 1.0 else 0.0) + abs(float(rng.normal(0, 1.4)))
            weather_code = 63 if rain_regime >= 5.0 else (61 if rain_regime > 0.3 else 1)

            weather_rows.append(
                {
                    "observed_hour_epoch": epoch_h,
                    "observed_at": dt_h.isoformat(),
                    "temp_c": round(temp_c, 2),
                    "precip_mm": round(rain_regime, 2),
                    "wind_speed_ms": round(wind_ms, 2),
                    "weather_code": weather_code,
                }
            )

        weather_by_hour = {w["observed_hour_epoch"]: w for w in weather_rows}

        # 2. Rail disruption episodes per rail line
        rail_rows: list[dict[str, Any]] = []
        active_incidents: dict[str, tuple[int, str, str, float, int]] = {}

        for b in range(total_buckets):
            dt_b = start_dt + timedelta(minutes=15 * b)
            epoch_b = int(dt_b.timestamp())
            hour_jst = dt_b.hour
            is_weekday = dt_b.weekday() < 5
            w = weather_by_hour[_floor_hour(epoch_b)]

            for rl in RAIL_LINES:
                if rl.line_id in active_incidents:
                    rem, st_ja, txt_ja, sev, furikae = active_incidents[rl.line_id]
                    if rem > 1:
                        active_incidents[rl.line_id] = (rem - 1, st_ja, txt_ja, sev, furikae)
                    else:
                        del active_incidents[rl.line_id]
                else:
                    rush_mult = 2.2 if (is_weekday and (7 <= hour_jst <= 9 or 17 <= hour_jst <= 20)) else 1.0
                    weather_mult = 2.4 if (w["precip_mm"] > 4.0 or w["wind_speed_ms"] > 7.5) else 1.0
                    p_incident = 0.0085 * rush_mult * weather_mult

                    if rng.random() < p_incident:
                        if w["precip_mm"] > 4.0 or w["wind_speed_ms"] > 7.5:
                            st_ja, tmpl = DISRUPTION_TEMPLATES[2]
                        else:
                            idx = int(rng.choice([0, 1, 3, 4], p=[0.22, 0.34, 0.28, 0.16]))
                            st_ja, tmpl = DISRUPTION_TEMPLATES[idx]
                        txt_ja = tmpl.format(line_ja=rl.line_name_ja, hub=rl.primary_hub)
                        parsed = parse_japanese_bulletin(txt_ja, st_ja)
                        duration_buckets = int(rng.integers(3, 7))
                        active_incidents[rl.line_id] = (
                            duration_buckets,
                            st_ja,
                            txt_ja,
                            parsed.severity_score,
                            parsed.has_furikae_yuso,
                        )

                if rl.line_id in active_incidents:
                    _, st_ja, txt_ja, _, _ = active_incidents[rl.line_id]
                else:
                    st_ja = "平常運転"
                    txt_ja = f"【{rl.line_name_ja}】現在、平常どおり運転しています。"

                parsed = parse_japanese_bulletin(txt_ja, st_ja)
                alert_id = hashlib.sha1(f"{rl.line_id}:{epoch_b}".encode()).hexdigest()[:16]
                rail_rows.append(
                    {
                        "alert_id": alert_id,
                        "observed_at": dt_b.isoformat(),
                        "observed_epoch": epoch_b,
                        "operator_name": rl.operator_name,
                        "line_id": rl.line_id,
                        "status_ja": st_ja,
                        "text_ja": txt_ja,
                        "is_disrupted": parsed.is_disrupted,
                        "parsed_cause": parsed.parsed_cause,
                        "has_furikae_yuso": parsed.has_furikae_yuso,
                        "severity_score": parsed.severity_score,
                    }
                )

        rail_sev_lookup: dict[tuple[str, int], tuple[float, int]] = {}
        for idx, row in enumerate(rail_rows):
            b_idx = idx // len(RAIL_LINES)
            rail_sev_lookup[(row["line_id"], b_idx)] = (row["severity_score"], row["has_furikae_yuso"])

        # 3. Toei Bus corridor telemetry with cross-modal cascade spillover
        bus_rows: list[dict[str, Any]] = []
        for br in BUS_ROUTES:
            ar_state = 0.0
            for b in range(total_buckets):
                dt_b = start_dt + timedelta(minutes=15 * b)
                epoch_b = int(dt_b.timestamp())
                hour_jst = dt_b.hour
                is_weekday = dt_b.weekday() < 5
                w = weather_by_hour[_floor_hour(epoch_b)]

                morning_rush = 145.0 * math.exp(-0.5 * ((hour_jst - 8.5) / 1.3) ** 2) if is_weekday else 35.0
                evening_rush = 175.0 * math.exp(-0.5 * ((hour_jst - 18.5) / 1.5) ** 2) if is_weekday else 65.0
                base_delay = 55.0 + morning_rush + evening_rush

                weather_delay = 14.0 * w["precip_mm"] + max(0.0, (w["wind_speed_ms"] - 5.0) * 6.0)

                cascade_delay = 0.0
                for lag_b, weight in [(1, 0.65), (2, 1.00), (3, 0.75)]:
                    if b - lag_b >= 0:
                        sev, furikae = rail_sev_lookup[(br.nearest_rail_line_id, b - lag_b)]
                        if sev > 0:
                            proximity_factor = max(0.5, 1.25 - br.hub_distance_km)
                            furikae_boost = 1.45 if furikae == 1 else 1.0
                            cascade_delay += weight * sev * 260.0 * proximity_factor * furikae_boost

                ar_state = 0.55 * ar_state + float(rng.normal(0, 22.0))
                mean_delay = max(10.0, base_delay + weather_delay + cascade_delay + ar_state)
                p90_delay = mean_delay * float(rng.uniform(1.25, 1.55)) + 30.0
                headway = br.baseline_headway_sec + 0.42 * (mean_delay - 60.0) + float(rng.normal(0, 28.0))
                headway = max(180.0, headway)
                occupancy = min(1.0, max(0.15, 0.35 + (mean_delay / 900.0) + (0.20 if cascade_delay > 80 else 0.0)))
                active_vehicles = max(3, int(round(3600 / br.baseline_headway_sec * float(rng.uniform(1.4, 1.9)))))

                obs_id = hashlib.sha1(f"{br.route_id}:{epoch_b}".encode()).hexdigest()[:16]
                bus_rows.append(
                    {
                        "obs_id": obs_id,
                        "observed_at": dt_b.isoformat(),
                        "observed_epoch": epoch_b,
                        "bucket_15m_epoch": _floor_15m(epoch_b),
                        "route_id": br.route_id,
                        "active_vehicles": active_vehicles,
                        "mean_delay_sec": round(mean_delay, 2),
                        "p90_delay_sec": round(p90_delay, 2),
                        "mean_headway_sec": round(headway, 2),
                        "occupancy_ratio": round(occupancy, 3),
                    }
                )

        cur = conn.cursor()
        cur.execute("DELETE FROM bus_telemetry")
        cur.execute("DELETE FROM rail_alerts")
        cur.execute("DELETE FROM weather_hourly")
        conn.commit()

        pd.DataFrame(weather_rows).to_sql("weather_hourly", conn, if_exists="append", index=False)
        pd.DataFrame(rail_rows).to_sql("rail_alerts", conn, if_exists="append", index=False)
        pd.DataFrame(bus_rows).to_sql("bus_telemetry", conn, if_exists="append", index=False)
        conn.commit()

        logger.info(
            "Inserted %d weather rows, %d rail status rows, and %d bus telemetry rows.",
            len(weather_rows),
            len(rail_rows),
            len(bus_rows),
        )
    finally:
        conn.close()


def poll_live_feeds(db_url: str | None = None) -> None:
    """Poll live ODPT TrainInformation, Toei Bus GTFS-Realtime, and Open-Meteo Tokyo weather."""
    conn = get_connection(db_url)
    try:
        init_schema(conn)
        cur = conn.cursor()
        now_jst = datetime.now(JST)
        epoch_now = int(now_jst.timestamp())
        bucket_15m = _floor_15m(epoch_now)
        hour_epoch = _floor_hour(epoch_now)

        # 1. Poll Tokyo Weather (Open-Meteo requires no API key)
        try:
            w_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 35.6895,
                    "longitude": 139.6917,
                    "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                    "wind_speed_unit": "ms",
                },
                timeout=15,
            )
            w_resp.raise_for_status()
            cur_w = w_resp.json().get("current", {})
            cur.execute(
                """
                INSERT INTO weather_hourly (observed_hour_epoch, observed_at, temp_c, precip_mm, wind_speed_ms, weather_code)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(observed_hour_epoch) DO UPDATE SET
                    temp_c=excluded.temp_c, precip_mm=excluded.precip_mm, wind_speed_ms=excluded.wind_speed_ms
                """,
                (
                    hour_epoch,
                    now_jst.isoformat(),
                    float(cur_w.get("temperature_2m", 16.0)),
                    float(cur_w.get("precipitation", 0.0)),
                    float(cur_w.get("wind_speed_10m", 3.0)),
                    int(cur_w.get("weather_code", 0)),
                ),
            )
            conn.commit()
            logger.info("Ingested live Tokyo weather snapshot.")
        except Exception as exc:
            logger.warning("Weather poll failed: %s", exc)

        # 2. Poll ODPT Rail Information
        if not ODPT_API_KEY:
            logger.warning("ODPT_API_KEY not set; skipping ODPT TrainInformation poll.")
        else:
            try:
                r_resp = requests.get(
                    ODPT_TRAIN_INFO_URL,
                    params={"acl:consumerKey": ODPT_API_KEY},
                    timeout=20,
                )
                r_resp.raise_for_status()
                items = r_resp.json()
                tracked_ids = {rl.line_id: rl for rl in RAIL_LINES}
                inserted = 0
                for item in items:
                    railway = item.get("odpt:railway", "")
                    if railway not in tracked_ids:
                        continue
                    status_obj = item.get("odpt:trainInformationStatus", {})
                    text_obj = item.get("odpt:trainInformationText", {})
                    status_ja = status_obj.get("ja", "平常運転") if isinstance(status_obj, dict) else str(status_obj)
                    text_ja = text_obj.get("ja", "") if isinstance(text_obj, dict) else str(text_obj)
                    parsed = parse_japanese_bulletin(text_ja, status_ja)
                    alert_id = hashlib.sha1(f"{railway}:{bucket_15m}".encode()).hexdigest()[:16]
                    cur.execute(
                        """
                        INSERT INTO rail_alerts (
                            alert_id, observed_at, observed_epoch, operator_name, line_id,
                            status_ja, text_ja, is_disrupted, parsed_cause, has_furikae_yuso, severity_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(alert_id) DO NOTHING
                        """,
                        (
                            alert_id,
                            now_jst.isoformat(),
                            epoch_now,
                            tracked_ids[railway].operator_name,
                            railway,
                            status_ja,
                            text_ja,
                            parsed.is_disrupted,
                            parsed.parsed_cause,
                            parsed.has_furikae_yuso,
                            parsed.severity_score,
                        ),
                    )
                    inserted += 1
                conn.commit()
                logger.info("Ingested %d ODPT rail status records.", inserted)
            except Exception as exc:
                logger.warning("ODPT TrainInformation poll failed: %s", exc)

        # 3. Poll Toei Bus GTFS-Realtime TripUpdate feed
        try:
            from google.transit import gtfs_realtime_pb2

            params = {"acl:consumerKey": ODPT_API_KEY} if ODPT_API_KEY else {}
            b_resp = requests.get(TOEI_TRIP_UPDATE_URL, params=params, timeout=20)
            b_resp.raise_for_status()
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(b_resp.content)

            route_delays: dict[str, list[float]] = {br.route_code: [] for br in BUS_ROUTES}
            for entity in feed.entity:
                if not entity.HasField("trip_update"):
                    continue
                tu = entity.trip_update
                r_id = tu.trip.route_id
                for code in route_delays:
                    if code in r_id:
                        delays = [
                            stu.arrival.delay
                            for stu in tu.stop_time_update
                            if stu.HasField("arrival") and stu.arrival.HasField("delay")
                        ]
                        if delays:
                            route_delays[code].append(float(np.mean(delays)))

            for br in BUS_ROUTES:
                vals = route_delays.get(br.route_code, [])
                if not vals:
                    continue
                mean_d = float(np.mean(vals))
                p90_d = float(np.percentile(vals, 90))
                obs_id = hashlib.sha1(f"{br.route_id}:{bucket_15m}".encode()).hexdigest()[:16]
                cur.execute(
                    """
                    INSERT INTO bus_telemetry (
                        obs_id, observed_at, observed_epoch, bucket_15m_epoch, route_id,
                        active_vehicles, mean_delay_sec, p90_delay_sec, mean_headway_sec, occupancy_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(obs_id) DO NOTHING
                    """,
                    (
                        obs_id,
                        now_jst.isoformat(),
                        epoch_now,
                        bucket_15m,
                        br.route_id,
                        len(vals),
                        round(mean_d, 2),
                        round(p90_d, 2),
                        float(br.baseline_headway_sec),
                        round(min(1.0, 0.4 + mean_d / 900.0), 3),
                    ),
                )
            conn.commit()
            logger.info("Ingested live Toei Bus GTFS-Realtime TripUpdate snapshot.")
        except Exception as exc:
            logger.warning("Toei Bus GTFS-Realtime poll skipped/failed: %s", exc)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokyo Multimodal Transit Ingestion Pipeline")
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--days", type=int, default=14, help="Number of days to generate in simulate mode")
    args = parser.parse_args()

    if args.mode == "simulate":
        generate_simulated_history(days=args.days)
    else:
        poll_live_feeds()


if __name__ == "__main__":
    main()
