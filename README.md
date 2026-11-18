# Tokyo Multimodal Transit Disruption & Cascade Predictor

**Real-Time Cross-Modal Delay Forecasting: Fusing ODPT Japanese Rail Bulletins & Toei Bus GTFS-Realtime Telemetry**

## Overview
While Tokyo's rail network maintains high baseline punctuality, acute service disruptions (*運転見合わせ* suspensions, *人身事故* track incidents, *車両点検* rolling-stock faults, and *強風* severe-wind restrictions) trigger immediate passenger spillover onto surface bus corridors—especially when operators activate **振替輸送** (*furikae yuso*, alternative transit transfer acceptance).

This system ingests structured bus telemetry (Toei Bus GTFS-Realtime), hourly meteorological observations (Open-Meteo Tokyo), and unstructured Japanese railway operational bulletins (Public Transportation Open Data Center / ODPT `odpt:TrainInformation`) into a normalized relational database, engineers lag-safe rolling features in SQL, and forecasts **30-minute downstream bus corridor delays and surge alerts** using gradient-boosted trees and bilingual NLP embeddings.

## System Architecture

```mermaid
flowchart LR
    subgraph Sources["External Data Sources"]
        ODPT["ODPT API<br/>(odpt:TrainInformation<br/>Japanese Rail Bulletins)"]
        GTFS["Toei Bus GTFS-RT<br/>(TripUpdate &<br/>VehiclePosition Protobuf)"]
        WX["Open-Meteo API<br/>(Tokyo Hourly Temp,<br/>Precip & Wind)"]
    end

    subgraph ETL["Ingestion & Bilingual NLP (src/ingest.py, src/nlp_parser.py)"]
        POLL["Scheduled Poller /<br/>14-Day Cascade Simulator"]
        NLP["Japanese Incident Extractor<br/>(Cause Taxonomy, 振替輸送 Flag,<br/>Severity Score & Embeddings)"]
    end

    subgraph DB["Relational Store (sql/schema.sql)"]
        SQL[(SQLite / PostgreSQL<br/>5 Normalized Tables)]
    end

    subgraph FE["SQL Feature Engineering (sql/features.sql)"]
        WIN["CTEs & Window Functions<br/>(Rolling 60m Delay/Headway,<br/>LAG/LEAD, 45m Hub Alert Window)"]
    end

    subgraph ML["Modeling & Delivery (src/train.py, app.py)"]
        XGB["Chronological 3-Tier<br/>XGBoost Ablation Pipeline"]
        UI["Streamlit Executive<br/>Decision Dashboard"]
    end

    ODPT --> POLL --> NLP --> SQL
    GTFS --> POLL
    WX --> POLL
    SQL --> WIN --> XGB --> UI
```

## Relational Database Schema (ERD)

```mermaid
erDiagram
    RAIL_LINES ||--o{ RAIL_ALERTS : "generates"
    RAIL_LINES ||--o{ BUS_ROUTES : "nearest_rail_line_id"
    BUS_ROUTES ||--o{ BUS_TELEMETRY : "records"

    RAIL_LINES {
        text line_id PK
        text operator_name
        text line_name_en
        text line_name_ja
        text primary_hub
        real hub_lat
        real hub_lon
    }

    BUS_ROUTES {
        text route_id PK
        text route_code
        text route_name_en
        text anchor_hub
        text nearest_rail_line_id FK
        real hub_distance_km
        int baseline_headway_sec
    }

    RAIL_ALERTS {
        text alert_id PK
        int observed_epoch
        text line_id FK
        text status_ja
        text text_ja
        int is_disrupted
        text parsed_cause
        int has_furikae_yuso
        real severity_score
    }

    BUS_TELEMETRY {
        text obs_id PK
        int bucket_15m_epoch
        text route_id FK
        int active_vehicles
        real mean_delay_sec
        real p90_delay_sec
        real mean_headway_sec
        real occupancy_ratio
    }

    WEATHER_HOURLY {
        int observed_hour_epoch PK
        real temp_c
        real precip_mm
        real wind_speed_ms
        int weather_code
    }
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Populate local SQLite database with 14 days of multimodal Tokyo transit history
python -m src.ingest --mode simulate --days 14

# 2. Run SQL feature engineering + chronological 3-tier ablation benchmark
python -m src.train

# 3. Run unit & integration tests
pytest -v

# 4. Launch the interactive 4-tab Streamlit dashboard
streamlit run app.py
```

## Live Data Collection (Optional)
1. Register for a free developer key at [developer.odpt.org](https://developer.odpt.org/).
2. Copy `.env.example` to `.env` and set `ODPT_API_KEY`.
3. Run `python -m src.ingest --mode live` or enable the GitHub Actions workflow in `.github/workflows/ingest.yml`.
