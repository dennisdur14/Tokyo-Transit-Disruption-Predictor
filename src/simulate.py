"""Synthetic test data with a known, planted answer.

The simulator exists to *test the pipeline*, not to produce findings about Tokyo. It
generates rail incidents, the Japanese bulletins an operator might publish about them, and
bus delays. Crucially, how much an incident delays buses is decided from hidden incident
attributes (phase, 振替輸送, a random per-incident size) -- never from the bulletin text or
the parser. The pipeline therefore has to recover the effect by reading the text, the same
way it would with real data.

Run it with planted_effect=False (a placebo) and rail features should stop helping. If they
still "help", information is leaking somewhere.

This module must not import src.nlp_parser (tests/test_simulation.py checks that the bus
delays don't depend on it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import BUCKET_SECONDS, BUS_ROUTES, RAIL_LINES, ROUTE_RAIL_LINKS

JST = timezone(timedelta(hours=9))

# ----------------------------------------------------------------------------------------
# The planted answer. None of these numbers are visible to the pipeline.
# ----------------------------------------------------------------------------------------
INCIDENTS_PER_LINE_PER_DAY = 0.35  # an assumption, not a measured rate
PHASE_EFFECT_SEC = {"delay": 45.0, "suspended": 150.0, "resumed": 70.0}  # per 15-min bucket
FURIKAE_MULTIPLIER = 1.6  # 振替輸送 pushes more rail riders onto buses
RELATION_STRENGTH = {"parallel": 1.0, "same_endpoints": 0.8, "shares_terminal": 0.5, "crosses": 0.35}
SPILLOVER_LAG_WEIGHTS = (0.3, 0.8, 1.0, 0.6)  # effect 0, 15, 30, 45 min after the rail state
INCIDENT_SIZE_SIGMA = 0.5  # lognormal spread of a hidden per-incident size factor

PHASES = ("normal", "delay", "suspended", "resumed")

# ----------------------------------------------------------------------------------------
# Text generation ingredients (real station names; phrasing modelled on real bulletins)
# ----------------------------------------------------------------------------------------
STATIONS = {
    "odpt.Railway:Toei.Oedo": ["都庁前", "新宿西口", "東新宿", "若松河田", "飯田橋", "春日", "本郷三丁目",
                               "上野御徒町", "両国", "門前仲町", "月島", "勝どき", "汐留", "大門",
                               "赤羽橋", "麻布十番", "六本木", "青山一丁目", "国立競技場", "中野坂上"],
    "odpt.Railway:Toei.Mita": ["目黒", "白金台", "白金高輪", "三田", "芝公園", "御成門", "内幸町",
                               "日比谷", "大手町", "神保町", "水道橋", "春日", "白山", "巣鴨", "高島平"],
    "odpt.Railway:TokyoMetro.Ginza": ["渋谷", "表参道", "外苑前", "青山一丁目", "赤坂見附", "溜池山王",
                                      "虎ノ門", "新橋", "銀座", "日本橋", "神田", "上野", "浅草"],
    "odpt.Railway:TokyoMetro.Marunouchi": ["荻窪", "東高円寺", "新中野", "中野坂上", "西新宿", "新宿",
                                           "四谷三丁目", "赤坂見附", "霞ケ関", "銀座", "東京", "大手町",
                                           "本郷三丁目", "後楽園", "茗荷谷", "池袋"],
    "odpt.Railway:JR-East.Yamanote": ["東京", "有楽町", "新橋", "浜松町", "品川", "大崎", "五反田", "目黒",
                                      "恵比寿", "渋谷", "原宿", "新宿", "高田馬場", "池袋", "大塚",
                                      "巣鴨", "日暮里", "上野", "秋葉原", "神田"],
    "odpt.Railway:Tokyu.DenEnToshi": ["渋谷", "池尻大橋", "三軒茶屋", "駒沢大学", "用賀", "二子玉川",
                                      "溝の口", "鷺沼", "たまプラーザ", "長津田", "中央林間"],
}

# cause -> (phrasings, probability the incident escalates to a suspension, (min, max) buckets suspended)
CAUSES = {
    "track_injury_accident": (["人身事故"], 0.85, (4, 8)),
    "rolling_stock_fault": (["車両点検", "車両故障", "ドア点検"], 0.15, (2, 3)),
    "signal_power_fault": (["信号確認", "信号トラブル", "停電"], 0.40, (2, 4)),
    "passenger_or_trespass": (["急病人救護", "お客様救護", "線路内立ち入り"], 0.20, (1, 3)),
    "congestion_knock_on": (["混雑", "他社線内での遅延"], 0.0, (1, 1)),
    "weather_severe": (["強風", "大雨"], 0.50, (2, 6)),
}
CAUSE_WEIGHTS_FAIR_WEATHER = {"track_injury_accident": 0.14, "rolling_stock_fault": 0.24,
                              "signal_power_fault": 0.16, "passenger_or_trespass": 0.32,
                              "congestion_knock_on": 0.14}
FURIKAE_OPERATORS = ["都営地下鉄", "都営バス", "東京メトロ", "JR線", "東急線", "京王線"]


@dataclass(frozen=True)
class SimConfig:
    start: datetime = datetime(2026, 10, 1, tzinfo=JST)  # fixed so every run is identical
    days: int = 28
    seed: int = 42
    planted_effect: bool = True


@dataclass
class Simulation:
    config: SimConfig
    weather: pd.DataFrame  # hour_epoch, temp_c, precip_mm, wind_speed_ms, weather_code
    rail: pd.DataFrame  # line_id, observed_epoch, status_ja, text_ja + truth_* columns
    bus: pd.DataFrame  # route_id, bucket_15m_epoch, n_vehicles, mean/p90 delay + truth_spillover_sec
    incidents: pd.DataFrame = field(default_factory=pd.DataFrame)


def simulate(cfg: SimConfig = SimConfig()) -> Simulation:
    # Independent random streams, so switching the planted effect off changes *only* the
    # spillover: same weather, same incidents, same bulletins, same bus noise.
    rng_weather, rng_incidents, rng_text, rng_bus = (
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(4)
    )
    times = pd.date_range(pd.Timestamp(cfg.start), periods=cfg.days * 96, freq="15min")
    epochs = np.array([int(t.timestamp()) for t in times])

    weather = _simulate_weather(rng_weather, cfg)
    bad_weather = _bad_weather_by_bucket(weather, epochs)
    timelines, incidents = _simulate_incidents(rng_incidents, times, bad_weather)
    rail = _write_bulletins(rng_text, epochs, timelines, incidents)
    bus = _simulate_bus(rng_bus, times, epochs, weather, timelines, cfg.planted_effect)
    return Simulation(cfg, weather, rail, bus, incidents)


# ----------------------------------------------------------------------------------------
# Weather
# ----------------------------------------------------------------------------------------
def _simulate_weather(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    rows, rain = [], 0.0
    start_hour = pd.Timestamp(cfg.start).floor("h")
    for h in range(cfg.days * 24 + 1):
        t = start_hour + pd.Timedelta(hours=h)
        rain = float(rng.uniform(2.0, 12.0)) if rng.random() < 0.04 else max(0.0, rain * 0.7 - 0.2)
        wind = 2.5 + (3.0 if rain > 1 else 0.0) + abs(float(rng.normal(0, 1.8)))
        rows.append({
            "hour_epoch": int(t.timestamp()),
            "temp_c": round(14 + 4 * np.sin((t.hour - 9) * np.pi / 12) + float(rng.normal(0, 1)), 1),
            "precip_mm": round(rain, 2),
            "wind_speed_ms": round(wind, 2),
            "weather_code": 63 if rain >= 5 else (61 if rain > 0.3 else 1),
        })
    return pd.DataFrame(rows)


def _bad_weather_by_bucket(weather: pd.DataFrame, epochs: np.ndarray) -> np.ndarray:
    w = weather.set_index("hour_epoch")
    hour = epochs - epochs % 3600
    return ((w.loc[hour, "precip_mm"].to_numpy() > 4.0) | (w.loc[hour, "wind_speed_ms"].to_numpy() > 8.5))


# ----------------------------------------------------------------------------------------
# Rail incidents: hidden truth
# ----------------------------------------------------------------------------------------
def _simulate_incidents(rng: np.random.Generator, times: pd.DatetimeIndex, bad_weather: np.ndarray):
    n = len(times)
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    weekday = times.day_of_week.to_numpy() < 5
    rush = weekday & (((hour >= 7) & (hour < 9.5)) | ((hour >= 17) & (hour < 20)))
    trains_running = ~((hour >= 1) & (hour < 5))

    timelines: dict[str, dict[str, np.ndarray]] = {}
    incidents: list[dict] = []
    for line in RAIL_LINES:
        tl = {
            "phase": np.zeros(n, dtype=int),  # index into PHASES
            "furikae": np.zeros(n, dtype=bool),
            "size": np.zeros(n),
            "incident": np.full(n, -1),
            "recovering": np.zeros(n, dtype=bool),  # first normal bucket after an incident
        }
        t = 0
        while t < n:
            hazard = INCIDENTS_PER_LINE_PER_DAY / 96 * (1.8 if rush[t] else 1.0) * (2.0 if bad_weather[t] else 1.0)
            if not trains_running[t] or rng.random() >= hazard:
                t += 1
                continue
            inc = _draw_incident(rng, line.line_id, t, bad_weather[t], len(incidents))
            for j, (phase, furikae) in enumerate(inc.pop("timeline")):
                if t + j < n:
                    tl["phase"][t + j] = PHASES.index(phase)
                    tl["furikae"][t + j] = furikae
                    tl["size"][t + j] = inc["size"]
                    tl["incident"][t + j] = inc["incident_id"]
            end = t + inc["n_buckets"]
            if end < n:
                tl["recovering"][end] = True
                tl["incident"][end] = inc["incident_id"]
            incidents.append(inc)
            t = end + 2  # at least 30 quiet minutes between incidents on one line
        timelines[line.line_id] = tl
    return timelines, pd.DataFrame(incidents)


def _draw_incident(rng, line_id: str, t: int, bad_weather: bool, incident_id: int) -> dict:
    if bad_weather and rng.random() < 0.6:
        cause = "weather_severe"
    else:
        names, probs = zip(*CAUSE_WEIGHTS_FAIR_WEATHER.items())
        cause = str(rng.choice(names, p=probs))
    phrasings, p_suspend, (lo, hi) = CAUSES[cause]
    suspends = rng.random() < p_suspend
    timeline: list[tuple[str, bool]] = []
    if suspends:
        if rng.random() < 0.3:
            timeline.append(("delay", False))  # starts as a delay, then escalates
        furikae = rng.random() < 0.7
        n_suspended = int(rng.integers(lo, hi + 1))
        announce_at = int(rng.integers(0, 2))  # 振替輸送 is often announced a little later
        timeline += [("suspended", furikae and k >= announce_at) for k in range(n_suspended)]
        timeline += [("resumed", furikae and k == 0) for k in range(int(rng.integers(1, 4)))]
    else:
        furikae = rng.random() < 0.05
        timeline += [("delay", furikae)] * int(rng.integers(1, 5))
    return {
        "incident_id": incident_id,
        "line_id": line_id,
        "start_bucket": t,
        "cause": cause,
        "cause_ja": str(rng.choice(phrasings)),
        "suspended": bool(suspends),
        "furikae": bool(any(f for _, f in timeline)),
        "size": float(rng.lognormal(0.0, INCIDENT_SIZE_SIGMA)),  # hidden: never written into text
        "n_buckets": len(timeline),
        "timeline": timeline,
    }


# ----------------------------------------------------------------------------------------
# Bulletins: what an operator would publish about the hidden state
# ----------------------------------------------------------------------------------------
def _write_bulletins(rng, epochs, timelines, incidents: pd.DataFrame) -> pd.DataFrame:
    by_id = incidents.set_index("incident_id") if len(incidents) else None
    rows = []
    for line in RAIL_LINES:
        tl = timelines[line.line_id]
        for i, epoch in enumerate(epochs):
            phase = PHASES[tl["phase"][i]]
            inc = by_id.loc[tl["incident"][i]] if tl["incident"][i] >= 0 else None
            status, text = _bulletin(rng, line, phase, bool(tl["furikae"][i]), inc, bool(tl["recovering"][i]), int(epoch))
            rows.append({
                "line_id": line.line_id,
                "observed_epoch": int(epoch) + 60,  # snapshot taken a minute into the bucket
                "status_ja": status,
                "text_ja": text,
                "truth_phase": phase,
                "truth_furikae": int(tl["furikae"][i]),
                "truth_cause": inc["cause"] if inc is not None and phase != "normal" else "normal",
            })
    return pd.DataFrame(rows)


def _clock(epoch: int, minutes_ahead: int) -> str:
    t = datetime.fromtimestamp(epoch, JST) + timedelta(minutes=minutes_ahead)
    return f"{t.hour}時{t.minute // 5 * 5:02d}分頃"


def _bulletin(rng, line, phase: str, furikae: bool, inc, recovering: bool, epoch: int) -> tuple[str, str]:
    pick = lambda options: str(options[int(rng.integers(len(options)))])  # noqa: E731
    stations = STATIONS[line.line_id]
    a, b = rng.choice(len(stations), size=2, replace=False)
    st1, st2 = stations[min(a, b)], stations[max(a, b)]

    if phase == "normal":
        if recovering and inc is not None and rng.random() < 0.5:
            return "", f"{inc['cause_ja']}の影響で遅れが出ていましたが、現在は平常どおり運転しています。"
        if line.operator == "Toei":  # real Toei wording, seen on the live feed
            return "", "現在、15分以上の遅延はありません。"
        return pick(["", "平常運転"]), pick(["平常どおり運転しています。", "現在、平常どおり運転しています。"])

    cause = inc["cause_ja"]
    # Operators don't always send a status header; then only the text says what's going on.
    send_header = rng.random() < 0.85
    if phase == "delay":
        header = pick(["遅延", "ダイヤ乱れ"])
        text = pick([
            f"{st1}駅で発生した{cause}の影響で、一部列車に{int(rng.integers(10, 31))}分程度の遅れが出ています。",
            f"{cause}の影響で、{line.name_ja}は遅れが出ています。",
            f"{st1}駅〜{st2}駅間での{cause}のため、ダイヤが乱れています。",
        ])
    elif phase == "suspended":
        header = "運転見合わせ"
        text = pick([
            f"{st1}駅での{cause}の影響で、{st1}駅〜{st2}駅間の運転を見合わせています。運転再開は{_clock(epoch, int(rng.integers(30, 91)))}を見込んでいます。",
            f"{cause}のため、全線で運転を見合わせています。",
            f"{st1}駅〜{st2}駅間で発生した{cause}の影響で、上下線で運転を見合わせています。",
        ])
    else:  # resumed
        header = pick(["運転再開", "遅延"])
        text = f"{cause}の影響で運転を見合わせていましたが、{_clock(epoch, -int(rng.integers(5, 15)))}に運転を再開しました。"
        if rng.random() < 0.6:
            text += pick(["現在も遅れが出ています。", "ダイヤが乱れています。"])

    if furikae:
        ops = "、".join(rng.choice(FURIKAE_OPERATORS, size=int(rng.integers(2, 4)), replace=False))
        text += pick([f"現在、{ops}で振替輸送を行っています。", "振替輸送を実施しています。"])
    elif phase == "suspended" and rng.random() < 0.3:
        text += "なお、振替輸送は行っておりません。"
    return (header if send_header else ""), text


# ----------------------------------------------------------------------------------------
# Bus delays: baseline + weather + noise + planted spillover from linked rail lines
# ----------------------------------------------------------------------------------------
def _simulate_bus(rng, times, epochs, weather, timelines, planted_effect: bool) -> pd.DataFrame:
    n = len(times)
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    weekday = times.day_of_week.to_numpy() < 5
    w = weather.set_index("hour_epoch").loc[epochs - epochs % 3600]
    weather_delay = 12 * w["precip_mm"].to_numpy() + np.maximum(0, w["wind_speed_ms"].to_numpy() - 6) * 5
    peak = np.where(
        weekday,
        130 * np.exp(-0.5 * ((hour - 8.5) / 1.3) ** 2) + 160 * np.exp(-0.5 * ((hour - 18) / 1.6) ** 2),
        60 * np.exp(-0.5 * ((hour - 14) / 3.0) ** 2),
    )
    in_service = ~((hour >= 0.5) & (hour < 5.5))  # Toei buses don't run in the small hours

    # Hidden rail "pressure" per line and bucket, before lags and route exposure.
    pressure = {}
    for line_id, tl in timelines.items():
        effect = np.array([PHASE_EFFECT_SEC.get(PHASES[p], 0.0) for p in tl["phase"]])
        pressure[line_id] = effect * np.where(tl["furikae"], FURIKAE_MULTIPLIER, 1.0) * tl["size"]

    rows = []
    for route in BUS_ROUTES:
        spill = np.zeros(n)
        for r_id, line_id, relation in ROUTE_RAIL_LINKS:
            if r_id != route.route_id:
                continue
            for lag, weight in enumerate(SPILLOVER_LAG_WEIGHTS):
                spill[lag:] += RELATION_STRENGTH[relation] * weight * pressure[line_id][: n - lag]
        if not planted_effect:
            spill[:] = 0.0

        route_scale = float(rng.uniform(0.8, 1.3))
        noise, ar = np.zeros(n), 0.0
        for i in range(n):
            ar = 0.6 * ar + float(rng.normal(0, 25))
            noise[i] = ar
        mean_delay = np.maximum(-60, 50 + route_scale * peak + weather_delay + noise + spill)
        p90 = mean_delay + 45 + 0.3 * np.maximum(mean_delay, 0) + rng.normal(0, 10, n)
        vehicles = rng.poisson(np.where((hour >= 7) & (hour < 21), 9, 4))
        kept = in_service & (rng.random(n) > 0.01) & (vehicles > 0)  # ~1% missed snapshots
        rows.append(pd.DataFrame({
            "route_id": route.route_id,
            "bucket_15m_epoch": epochs[kept] - epochs[kept] % BUCKET_SECONDS,
            "n_vehicles": vehicles[kept],
            "mean_delay_sec": np.round(mean_delay[kept], 1),
            "p90_delay_sec": np.round(p90[kept], 1),
            "truth_spillover_sec": np.round(spill[kept], 1),
        }))
    return pd.concat(rows, ignore_index=True)
