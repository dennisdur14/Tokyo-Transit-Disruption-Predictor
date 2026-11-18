-- Relational schema (SQLite 3.28+; uses window functions with RANGE frames).
-- Reference tables describe the network; fact tables hold what was observed over time.

CREATE TABLE IF NOT EXISTS rail_lines (
    line_id    TEXT PRIMARY KEY,          -- ODPT railway ID, e.g. odpt.Railway:Toei.Oedo
    operator   TEXT NOT NULL,
    name_en    TEXT NOT NULL,
    name_ja    TEXT NOT NULL,
    needs_key  INTEGER NOT NULL DEFAULT 1 -- 0 = available from ODPT without an API key
);

CREATE TABLE IF NOT EXISTS bus_routes (
    route_id        TEXT PRIMARY KEY,
    route_code      TEXT NOT NULL,        -- Toei's public code, e.g. 都01
    gtfs_short_name TEXT NOT NULL,        -- matching route_short_name in Toei's GTFS (NFKC)
    name_en         TEXT NOT NULL,
    name_ja         TEXT NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL
);

-- Many-to-many: a bus route can be exposed to several rail lines and vice versa.
CREATE TABLE IF NOT EXISTS route_rail_links (
    route_id TEXT NOT NULL REFERENCES bus_routes(route_id),
    line_id  TEXT NOT NULL REFERENCES rail_lines(line_id),
    relation TEXT NOT NULL,               -- parallel / same_endpoints / crosses / shares_terminal
    PRIMARY KEY (route_id, line_id)
);

-- One row per rail line per observation (a status snapshot), with the parser's reading of it.
CREATE TABLE IF NOT EXISTS rail_alerts (
    alert_id         TEXT PRIMARY KEY,
    observed_epoch   INTEGER NOT NULL,    -- when we saw this status (unix seconds)
    line_id          TEXT NOT NULL REFERENCES rail_lines(line_id),
    status_ja        TEXT NOT NULL,       -- short status header ('' when the operator sends none)
    text_ja          TEXT NOT NULL,       -- free-text bulletin
    status_category  TEXT NOT NULL,       -- header only: normal / delay / suspended / resumed / other
    is_disrupted     INTEGER NOT NULL,    -- parser output from here down
    parsed_cause     TEXT NOT NULL,
    is_suspended     INTEGER NOT NULL,
    is_resumed       INTEGER NOT NULL,
    has_furikae_yuso INTEGER NOT NULL,
    severity_score   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rail_alerts_line_epoch ON rail_alerts(line_id, observed_epoch);

-- One row per bus route per 15-minute bucket, aggregated from individual vehicles.
CREATE TABLE IF NOT EXISTS bus_telemetry (
    route_id         TEXT NOT NULL REFERENCES bus_routes(route_id),
    bucket_15m_epoch INTEGER NOT NULL,    -- bucket start (unix seconds)
    n_vehicles       INTEGER NOT NULL,    -- vehicles that contributed a delay estimate
    mean_delay_sec   REAL NOT NULL,       -- schedule deviation (positive = late)
    p90_delay_sec    REAL NOT NULL,
    PRIMARY KEY (route_id, bucket_15m_epoch)
);

CREATE TABLE IF NOT EXISTS weather_hourly (
    hour_epoch    INTEGER PRIMARY KEY,
    temp_c        REAL NOT NULL,
    precip_mm     REAL NOT NULL,          -- mm per hour
    wind_speed_ms REAL NOT NULL,
    weather_code  INTEGER NOT NULL
);

-- Records where the data came from (simulated vs collected) and with which settings.
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
