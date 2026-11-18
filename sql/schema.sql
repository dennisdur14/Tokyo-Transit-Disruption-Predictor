-- Normalized Relational Schema for Tokyo Multimodal Transit Disruption & Cascade Predictor
-- Compatible with SQLite 3.35+ and PostgreSQL 14+

CREATE TABLE IF NOT EXISTS rail_lines (
    line_id TEXT PRIMARY KEY,
    operator_name TEXT NOT NULL,
    line_name_en TEXT NOT NULL,
    line_name_ja TEXT NOT NULL,
    primary_hub TEXT NOT NULL,
    hub_lat REAL NOT NULL,
    hub_lon REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bus_routes (
    route_id TEXT PRIMARY KEY,
    route_code TEXT NOT NULL,
    route_name_en TEXT NOT NULL,
    route_name_ja TEXT NOT NULL,
    anchor_hub TEXT NOT NULL,
    nearest_rail_line_id TEXT NOT NULL,
    hub_distance_km REAL NOT NULL,
    baseline_headway_sec INTEGER NOT NULL,
    FOREIGN KEY (nearest_rail_line_id) REFERENCES rail_lines(line_id)
);

CREATE TABLE IF NOT EXISTS rail_alerts (
    alert_id TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL,
    observed_epoch INTEGER NOT NULL,
    operator_name TEXT NOT NULL,
    line_id TEXT NOT NULL,
    status_ja TEXT NOT NULL,
    text_ja TEXT NOT NULL,
    is_disrupted INTEGER NOT NULL DEFAULT 0,
    parsed_cause TEXT NOT NULL DEFAULT 'normal',
    has_furikae_yuso INTEGER NOT NULL DEFAULT 0,
    severity_score REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (line_id) REFERENCES rail_lines(line_id)
);

CREATE INDEX IF NOT EXISTS idx_rail_alerts_line_epoch ON rail_alerts(line_id, observed_epoch);

CREATE TABLE IF NOT EXISTS bus_telemetry (
    obs_id TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL,
    observed_epoch INTEGER NOT NULL,
    bucket_15m_epoch INTEGER NOT NULL,
    route_id TEXT NOT NULL,
    active_vehicles INTEGER NOT NULL,
    mean_delay_sec REAL NOT NULL,
    p90_delay_sec REAL NOT NULL,
    mean_headway_sec REAL NOT NULL,
    occupancy_ratio REAL NOT NULL,
    FOREIGN KEY (route_id) REFERENCES bus_routes(route_id)
);

CREATE INDEX IF NOT EXISTS idx_bus_telemetry_route_bucket ON bus_telemetry(route_id, bucket_15m_epoch);

CREATE TABLE IF NOT EXISTS weather_hourly (
    observed_hour_epoch INTEGER PRIMARY KEY,
    observed_at TEXT NOT NULL,
    temp_c REAL NOT NULL,
    precip_mm REAL NOT NULL,
    wind_speed_ms REAL NOT NULL,
    weather_code INTEGER NOT NULL
);
