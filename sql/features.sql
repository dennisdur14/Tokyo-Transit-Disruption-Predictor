-- Multimodal Feature Engineering Pipeline (CTEs + Window Functions + Cross-Modal Spatial-Temporal Joins)
-- Produces 15-minute corridor-level modeling rows predicting downstream bus delay surges 30 minutes ahead.

WITH bus_ordered AS (
    SELECT
        bt.bucket_15m_epoch,
        bt.observed_at,
        bt.route_id,
        br.route_code,
        br.anchor_hub,
        br.nearest_rail_line_id,
        br.hub_distance_km,
        br.baseline_headway_sec,
        bt.active_vehicles,
        bt.mean_delay_sec,
        bt.p90_delay_sec,
        bt.mean_headway_sec,
        bt.occupancy_ratio,
        -- Rolling window statistics over past 45 minutes (3 prior 15-min buckets + current)
        AVG(bt.mean_delay_sec) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_60m_delay_sec,
        AVG(bt.mean_headway_sec) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_60m_headway_sec,
        LAG(bt.mean_delay_sec, 1) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
        ) AS lag_15m_delay_sec,
        LAG(bt.mean_delay_sec, 2) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
        ) AS lag_30m_delay_sec,
        -- Future target: corridor bus delay 30 minutes (2 buckets) ahead
        LEAD(bt.mean_delay_sec, 2) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
        ) AS target_delay_30m_sec
    FROM bus_telemetry bt
    JOIN bus_routes br ON bt.route_id = br.route_id
),
rail_windowed AS (
    SELECT
        bo.bucket_15m_epoch,
        bo.route_id,
        COALESCE(MAX(ra.is_disrupted), 0) AS hub_rail_disrupted,
        COALESCE(MAX(ra.severity_score), 0.0) AS hub_max_rail_severity,
        COALESCE(MAX(ra.has_furikae_yuso), 0) AS hub_furikae_active,
        COALESCE(SUM(ra.is_disrupted), 0) AS hub_active_alert_count,
        -- Concatenate active Japanese alert texts within the preceding 45 minutes for NLP embedding
        COALESCE(GROUP_CONCAT(CASE WHEN ra.is_disrupted = 1 THEN ra.text_ja END, ' | '), '') AS active_alerts_ja,
        COALESCE(MAX(CASE WHEN ra.is_disrupted = 1 THEN ra.parsed_cause END), 'normal') AS dominant_rail_cause
    FROM bus_ordered bo
    LEFT JOIN rail_alerts ra
        ON ra.line_id = bo.nearest_rail_line_id
        AND ra.observed_epoch BETWEEN (bo.bucket_15m_epoch - 2700) AND bo.bucket_15m_epoch
    GROUP BY bo.bucket_15m_epoch, bo.route_id
)
SELECT
    bo.bucket_15m_epoch,
    bo.observed_at,
    bo.route_id,
    bo.route_code,
    bo.anchor_hub,
    bo.nearest_rail_line_id,
    bo.hub_distance_km,
    bo.active_vehicles,
    bo.mean_delay_sec,
    bo.p90_delay_sec,
    bo.mean_headway_sec,
    (bo.mean_headway_sec - bo.baseline_headway_sec) AS headway_drift_sec,
    bo.occupancy_ratio,
    bo.rolling_60m_delay_sec,
    bo.rolling_60m_headway_sec,
    COALESCE(bo.lag_15m_delay_sec, bo.mean_delay_sec) AS lag_15m_delay_sec,
    COALESCE(bo.lag_30m_delay_sec, bo.mean_delay_sec) AS lag_30m_delay_sec,
    (bo.mean_delay_sec - COALESCE(bo.lag_15m_delay_sec, bo.mean_delay_sec)) AS delay_velocity_15m,
    -- Weather join on floored hour epoch
    COALESCE(wh.temp_c, 16.0) AS temp_c,
    COALESCE(wh.precip_mm, 0.0) AS precip_mm,
    COALESCE(wh.wind_speed_ms, 3.0) AS wind_speed_ms,
    -- Rail disruption & Japanese NLP features
    rw.hub_rail_disrupted,
    rw.hub_max_rail_severity,
    rw.hub_furikae_active,
    rw.hub_active_alert_count,
    rw.dominant_rail_cause,
    rw.active_alerts_ja,
    -- Target variables
    bo.target_delay_30m_sec,
    CASE
        WHEN (bo.target_delay_30m_sec - bo.rolling_60m_delay_sec) >= 180.0
             OR bo.target_delay_30m_sec >= 420.0
        THEN 1 ELSE 0
    END AS target_surge_30m
FROM bus_ordered bo
JOIN rail_windowed rw
    ON bo.bucket_15m_epoch = rw.bucket_15m_epoch
    AND bo.route_id = rw.route_id
LEFT JOIN weather_hourly wh
    ON wh.observed_hour_epoch = (bo.bucket_15m_epoch - (bo.bucket_15m_epoch % 3600))
WHERE bo.target_delay_30m_sec IS NOT NULL
ORDER BY bo.bucket_15m_epoch ASC, bo.route_id ASC;
