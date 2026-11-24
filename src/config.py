"""Paths, data-source URLs, and the hand-made route/rail reference data."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

try:  # optional: lets you keep ODPT_API_KEY in a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT_DIR / "sql"
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# os.getenv returns "" (not the default) when a variable exists but is empty -- which is
# what an unset GitHub secret looks like -- so fall back with `or`, not getenv's default.
DB_PATH = Path(os.getenv("TRANSIT_DB_PATH") or DATA_DIR / "tokyo_transit.db")
ODPT_API_KEY = os.getenv("ODPT_API_KEY") or ""

# --- Data sources (checked against the live services on 2026-11-18) ---
# Keyless ODPT endpoints: Toei bus positions, Toei bus timetable, Toei rail status.
TOEI_BUS_POSITIONS_URL = "https://api-public.odpt.org/api/v4/gtfs/realtime/ToeiBus"
TOEI_BUS_GTFS_URL = "https://api-public.odpt.org/api/v4/files/Toei/data/ToeiBus-GTFS.zip"
ODPT_PUBLIC_TRAIN_INFO_URL = "https://api-public.odpt.org/api/v4/odpt:TrainInformation"
# Needs a free developer key from developer.odpt.org; adds Tokyo Metro's lines (and a few
# others). JR East and Tokyu are NOT in this feed (checked with a key on 2026-11-24).
ODPT_TRAIN_INFO_URL = "https://api.odpt.org/api/v4/odpt:TrainInformation"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TOKYO_LAT, TOKYO_LON = 35.6895, 139.6917

BUCKET_SECONDS = 900  # 15-minute buckets
HORIZON_SECONDS = 1800  # forecast the bucket that starts 30 minutes later


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@dataclass(frozen=True)
class RailLine:
    line_id: str  # ODPT railway ID
    operator: str
    name_en: str
    name_ja: str
    needs_key: bool  # True = only available from the keyed ODPT endpoint


@dataclass(frozen=True)
class BusRoute:
    route_id: str
    route_code: str  # Toei's public route code
    gtfs_short_name: str  # route_short_name in Toei's GTFS after NFKC normalisation
    name_en: str
    name_ja: str
    lat: float  # centroid of the route's stops (for the map)
    lon: float


RAIL_LINES: list[RailLine] = [
    RailLine("odpt.Railway:Toei.Oedo", "Toei", "Toei Oedo Line", "都営大江戸線", False),
    RailLine("odpt.Railway:Toei.Mita", "Toei", "Toei Mita Line", "都営三田線", False),
    RailLine("odpt.Railway:TokyoMetro.Ginza", "TokyoMetro", "Tokyo Metro Ginza Line", "東京メトロ銀座線", True),
    RailLine("odpt.Railway:TokyoMetro.Marunouchi", "TokyoMetro", "Tokyo Metro Marunouchi Line", "東京メトロ丸ノ内線", True),
    RailLine("odpt.Railway:TokyoMetro.Hibiya", "TokyoMetro", "Tokyo Metro Hibiya Line", "東京メトロ日比谷線", True),
    RailLine("odpt.Railway:TokyoMetro.Namboku", "TokyoMetro", "Tokyo Metro Namboku Line", "東京メトロ南北線", True),
]

# Route names and coordinates were checked against Toei's published GTFS timetable.
BUS_ROUTES: list[BusRoute] = [
    BusRoute("Toei.To01", "都01", "都01(T01)", "Shibuya Sta. - Roppongi - Shimbashi Sta.",
             "渋谷駅前〜六本木駅前〜新橋駅前", 35.6643, 139.7343),
    BusRoute("Toei.Shuku74", "宿74", "宿74", "Shinjuku Sta. West - Higashi-Shinjuku - Tokyo Women's Medical Univ.",
             "新宿駅西口〜東新宿駅前〜東京女子医大前", 35.6978, 139.7112),
    BusRoute("Toei.Shuku91", "宿91", "宿91", "Shinjuku Sta. West - Nakano-Sakaue - Shin-Daita Sta.",
             "新宿駅西口〜中野坂上〜新代田駅前", 35.6886, 139.6666),
    BusRoute("Toei.Shina97", "品97", "品97", "Shinagawa Sta. - Aoyama-Itchome - Shinjuku Sta. West",
             "品川駅高輪口〜青山一丁目駅前〜新宿駅西口", 35.6598, 139.7248),
    BusRoute("Toei.To02", "都02", "都02", "Otsuka Sta. - Kasuga - Okachimachi - Kinshicho Sta.",
             "大塚駅前〜春日駅前〜御徒町駅前〜錦糸町駅前", 35.7092, 139.7710),
    BusRoute("Toei.Shibu88", "渋88", "渋88", "Shibuya Sta. - Roppongi - Kamiyacho - Shimbashi Sta.",
             "渋谷駅前〜六本木駅前〜神谷町駅前〜新橋駅前", 35.6631, 139.7321),
    BusRoute("Toei.To05-1", "都05-1", "都05-1", "Tokyo Sta. Marunouchi South - Ginza - Kachidoki - Harumi Futo",
             "東京駅丸の内南口〜銀座四丁目〜勝どき駅前〜晴海埠頭", 35.6639, 139.7719),
    BusRoute("Toei.Hashi86", "橋86", "橋86", "Shimbashi Sta. - Azabu-Juban - Shirokanedai - Meguro Sta.",
             "新橋駅前〜麻布十番駅前〜白金台五丁目〜目黒駅前", 35.6518, 139.7352),
]

# Which rail lines could plausibly push riders onto each bus route. This is a hypothesis
# read off the routes' real stop lists, not a measured fact -- the real-data analysis is
# what tests it. Relations:
#   parallel        - runs alongside the line for several stations
#   same_endpoints  - connects the same two hubs as the line
#   crosses         - meets the line at one station mid-route
#   shares_terminal - starts or ends at one of the line's stations
ROUTE_RAIL_LINKS: list[tuple[str, str, str]] = [
    ("Toei.To01", "odpt.Railway:TokyoMetro.Ginza", "same_endpoints"),
    ("Toei.To01", "odpt.Railway:Toei.Oedo", "crosses"),  # Roppongi
    ("Toei.To01", "odpt.Railway:TokyoMetro.Hibiya", "crosses"),  # Roppongi
    ("Toei.To01", "odpt.Railway:TokyoMetro.Namboku", "crosses"),  # Tameike, Roppongi-itchome
    ("Toei.Shuku74", "odpt.Railway:Toei.Oedo", "parallel"),  # Higashi-Shinjuku, Wakamatsu-Kawada
    ("Toei.Shuku91", "odpt.Railway:TokyoMetro.Marunouchi", "parallel"),  # Nakano-Sakaue .. Higashi-Koenji
    ("Toei.Shuku91", "odpt.Railway:Toei.Oedo", "crosses"),  # Nakano-Sakaue
    ("Toei.Shina97", "odpt.Railway:TokyoMetro.Marunouchi", "parallel"),  # Yotsuya-sanchome .. Shinjuku
    ("Toei.Shina97", "odpt.Railway:Toei.Oedo", "crosses"),  # Aoyama-itchome
    ("Toei.Shina97", "odpt.Railway:TokyoMetro.Hibiya", "crosses"),  # Hiroo
    ("Toei.To02", "odpt.Railway:TokyoMetro.Marunouchi", "parallel"),  # Shin-Otsuka .. Hongo-sanchome
    ("Toei.To02", "odpt.Railway:Toei.Oedo", "parallel"),  # Kasuga .. Kuramae
    ("Toei.To02", "odpt.Railway:Toei.Mita", "crosses"),  # Kasuga
    ("Toei.To02", "odpt.Railway:TokyoMetro.Ginza", "crosses"),  # Ueno-hirokoji
    ("Toei.Shibu88", "odpt.Railway:TokyoMetro.Ginza", "same_endpoints"),
    ("Toei.Shibu88", "odpt.Railway:TokyoMetro.Hibiya", "parallel"),  # Roppongi .. Kamiyacho .. Toranomon
    ("Toei.Shibu88", "odpt.Railway:Toei.Oedo", "crosses"),  # Roppongi
    ("Toei.To05-1", "odpt.Railway:Toei.Oedo", "parallel"),  # Tsukiji .. Kachidoki
    ("Toei.To05-1", "odpt.Railway:TokyoMetro.Hibiya", "parallel"),  # Ginza .. Tsukiji
    ("Toei.To05-1", "odpt.Railway:TokyoMetro.Marunouchi", "shares_terminal"),  # Tokyo
    ("Toei.Hashi86", "odpt.Railway:Toei.Mita", "parallel"),  # Meguro, Shirokanedai, Onarimon
    ("Toei.Hashi86", "odpt.Railway:TokyoMetro.Namboku", "parallel"),  # Meguro .. Shirokanedai .. Azabu-Juban
    ("Toei.Hashi86", "odpt.Railway:Toei.Oedo", "parallel"),  # Azabu-Juban, Akabanebashi
    ("Toei.Hashi86", "odpt.Railway:TokyoMetro.Hibiya", "crosses"),  # Kamiyacho, Hiroo
]
