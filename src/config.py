from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT_DIR / "sql"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ODPT_API_KEY = os.getenv("ODPT_API_KEY", "")
ODPT_TRAIN_INFO_URL = os.getenv(
    "ODPT_TRAIN_INFO_URL",
    "https://api.odpt.org/api/v4/odpt:TrainInformation",
)
TOEI_VEHICLE_POS_URL = os.getenv(
    "TOEI_VEHICLE_POS_URL",
    "https://api-public.odpt.org/api/v4/gtfs/realtime/ToeiBus_VehiclePosition",
)
TOEI_TRIP_UPDATE_URL = os.getenv(
    "TOEI_TRIP_UPDATE_URL",
    "https://api-public.odpt.org/api/v4/gtfs/realtime/ToeiBus_TripUpdate",
)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'tokyo_transit.db'}")


def get_connection(db_url: str | None = None):
    """Return a DB-API connection (sqlite3 locally or SQLAlchemy/psycopg2 for PostgreSQL)."""
    url = db_url or DATABASE_URL
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    from sqlalchemy import create_engine

    engine = create_engine(url, future=True)
    return engine.raw_connection()


@dataclass(frozen=True)
class RailLineSpec:
    line_id: str
    operator_name: str
    line_name_en: str
    line_name_ja: str
    primary_hub: str
    hub_lat: float
    hub_lon: float


@dataclass(frozen=True)
class BusRouteSpec:
    route_id: str
    route_code: str
    route_name_en: str
    route_name_ja: str
    anchor_hub: str
    nearest_rail_line_id: str
    hub_distance_km: float
    baseline_headway_sec: int


RAIL_LINES: list[RailLineSpec] = [
    RailLineSpec(
        line_id="odpt.Railway:JR-East.Yamanote",
        operator_name="JR-East",
        line_name_en="JR Yamanote Line",
        line_name_ja="JR山手線",
        primary_hub="Shinjuku",
        hub_lat=35.6896,
        hub_lon=139.7006,
    ),
    RailLineSpec(
        line_id="odpt.Railway:JR-East.ChuoRapid",
        operator_name="JR-East",
        line_name_en="JR Chuo Rapid Line",
        line_name_ja="JR中央線快速",
        primary_hub="Shinjuku",
        hub_lat=35.6909,
        hub_lon=139.7003,
    ),
    RailLineSpec(
        line_id="odpt.Railway:Odakyu.Odawara",
        operator_name="Odakyu",
        line_name_en="Odakyu Odawara Line",
        line_name_ja="小田急小田原線",
        primary_hub="Shinjuku-West",
        hub_lat=35.6889,
        hub_lon=139.6987,
    ),
    RailLineSpec(
        line_id="odpt.Railway:Tokyu.DenEnToshi",
        operator_name="Tokyu",
        line_name_en="Tokyu Den-en-toshi Line",
        line_name_ja="東急田園都市線",
        primary_hub="Shibuya",
        hub_lat=35.6580,
        hub_lon=139.7016,
    ),
    RailLineSpec(
        line_id="odpt.Railway:TokyoMetro.Ginza",
        operator_name="TokyoMetro",
        line_name_en="Tokyo Metro Ginza Line",
        line_name_ja="東京メトロ銀座線",
        primary_hub="Shimbashi",
        hub_lat=35.6664,
        hub_lon=139.7583,
    ),
    RailLineSpec(
        line_id="odpt.Railway:Toei.Oedo",
        operator_name="Toei",
        line_name_en="Toei Oedo Line",
        line_name_ja="都営大江戸線",
        primary_hub="Roppongi",
        hub_lat=35.6628,
        hub_lon=139.7314,
    ),
    RailLineSpec(
        line_id="odpt.Railway:JR-East.KeihinTohoku",
        operator_name="JR-East",
        line_name_en="JR Keihin-Tohoku Line",
        line_name_ja="JR京浜東北線",
        primary_hub="Shinagawa",
        hub_lat=35.6285,
        hub_lon=139.7388,
    ),
    RailLineSpec(
        line_id="odpt.Railway:TokyoMetro.Marunouchi",
        operator_name="TokyoMetro",
        line_name_en="Tokyo Metro Marunouchi Line",
        line_name_ja="東京メトロ丸ノ内線",
        primary_hub="Tokyo",
        hub_lat=35.6812,
        hub_lon=139.7671,
    ),
]

BUS_ROUTES: list[BusRouteSpec] = [
    BusRouteSpec(
        route_id="Toei.To01",
        route_code="都01",
        route_name_en="Shibuya Sta. - Roppongi - Shimbashi Sta.",
        route_name_ja="渋谷駅前〜六本木〜新橋駅前",
        anchor_hub="Shibuya",
        nearest_rail_line_id="odpt.Railway:Tokyu.DenEnToshi",
        hub_distance_km=0.18,
        baseline_headway_sec=360,
    ),
    BusRouteSpec(
        route_id="Toei.Shuku74",
        route_code="宿74",
        route_name_en="Shinjuku Sta. West - Tokyo Women's Medical Univ.",
        route_name_ja="新宿駅西口〜東京女子医大前",
        anchor_hub="Shinjuku-West",
        nearest_rail_line_id="odpt.Railway:Odakyu.Odawara",
        hub_distance_km=0.12,
        baseline_headway_sec=480,
    ),
    BusRouteSpec(
        route_id="Toei.Shuku91",
        route_code="宿91",
        route_name_en="Shinjuku Sta. West - Shin-Daita Sta.",
        route_name_ja="新宿駅西口〜新代田駅前",
        anchor_hub="Shinjuku",
        nearest_rail_line_id="odpt.Railway:JR-East.ChuoRapid",
        hub_distance_km=0.22,
        baseline_headway_sec=540,
    ),
    BusRouteSpec(
        route_id="Toei.Shina97",
        route_code="品97",
        route_name_en="Shinagawa Sta. Takanawa - Shinjuku Sta. West",
        route_name_ja="品川駅高輪口〜新宿駅西口",
        anchor_hub="Shinagawa",
        nearest_rail_line_id="odpt.Railway:JR-East.KeihinTohoku",
        hub_distance_km=0.25,
        baseline_headway_sec=600,
    ),
    BusRouteSpec(
        route_id="Toei.To02",
        route_code="都02",
        route_name_en="Otsuka Sta. - Kasuga - Kinshicho Sta.",
        route_name_ja="大塚駅前〜春日駅前〜錦糸町駅前",
        anchor_hub="Shinjuku",
        nearest_rail_line_id="odpt.Railway:JR-East.Yamanote",
        hub_distance_km=0.35,
        baseline_headway_sec=420,
    ),
    BusRouteSpec(
        route_id="Toei.Shibu88",
        route_code="渋88",
        route_name_en="Shibuya Sta. - Azabu-Juban - Shimbashi Sta.",
        route_name_ja="渋谷駅前〜麻布十番駅前〜新橋駅前",
        anchor_hub="Roppongi",
        nearest_rail_line_id="odpt.Railway:Toei.Oedo",
        hub_distance_km=0.20,
        baseline_headway_sec=540,
    ),
    BusRouteSpec(
        route_id="Toei.To05",
        route_code="都05-1",
        route_name_en="Tokyo Sta. Marunouchi South - Kachidoki - Harumi",
        route_name_ja="東京駅丸の内南口〜勝どき駅前〜晴海埠頭",
        anchor_hub="Tokyo",
        nearest_rail_line_id="odpt.Railway:TokyoMetro.Marunouchi",
        hub_distance_km=0.15,
        baseline_headway_sec=360,
    ),
    BusRouteSpec(
        route_id="Toei.Hashi86",
        route_code="橋86",
        route_name_en="Meguro Sta. - Azabu-Juban - Shimbashi Sta.",
        route_name_ja="目黒駅前〜麻布十番駅前〜新橋駅前",
        anchor_hub="Shimbashi",
        nearest_rail_line_id="odpt.Railway:TokyoMetro.Ginza",
        hub_distance_km=0.24,
        baseline_headway_sec=600,
    ),
]
