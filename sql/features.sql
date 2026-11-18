-- Modeling table: one row per (bus route, 15-minute bucket).
-- Target: the route's mean delay in the bucket that starts 30 minutes later.
-- A forecast is made at the END of the current bucket, so every feature below uses only
-- data observed before then (tests/test_features.py checks this by altering future data).

WITH bus AS (
    SELECT
        bt.route_id,
        bt.bucket_15m_epoch,
        bt.n_vehicles,
        bt.mean_delay_sec,
        bt.p90_delay_sec,
        -- Mean delay over the last hour. RANGE (not ROWS) makes the window time-based, so a
        -- missing bucket can't silently stretch it to cover more than an hour.
        AVG(bt.mean_delay_sec) OVER (
            PARTITION BY bt.route_id
            ORDER BY bt.bucket_15m_epoch
            RANGE BETWEEN 2700 PRECEDING AND CURRENT ROW
        ) AS rolling_60m_delay_sec,
        -- LAG/LEAD return the neighbouring *row*, which is only 15/30 minutes away when no
        -- bucket is missing (buses don't run at night), so check the timestamp first.
        CASE WHEN LAG(bt.bucket_15m_epoch, 1) OVER w = bt.bucket_15m_epoch - 900
             THEN LAG(bt.mean_delay_sec, 1) OVER w END AS lag_15m_delay_sec,
        CASE WHEN LAG(bt.bucket_15m_epoch, 2) OVER w = bt.bucket_15m_epoch - 1800
             THEN LAG(bt.mean_delay_sec, 2) OVER w END AS lag_30m_delay_sec,
        -- The label. LEAD looks into the future, so it is used here and nowhere else.
        CASE WHEN LEAD(bt.bucket_15m_epoch, 2) OVER w = bt.bucket_15m_epoch + 1800
             THEN LEAD(bt.mean_delay_sec, 2) OVER w END AS target_delay_30m_sec
    FROM bus_telemetry bt
    WINDOW w AS (PARTITION BY bt.route_id ORDER BY bt.bucket_15m_epoch)
),

-- Every bulletin from each route's linked rail lines seen during the current bucket or the
-- 45 minutes before it.
rail_recent AS (
    SELECT
        b.route_id,
        b.bucket_15m_epoch,
        l.relation IN ('parallel', 'same_endpoints') AS strong_link,
        ra.observed_epoch,
        ra.text_ja,
        ra.status_category,
        ra.is_disrupted,
        ra.parsed_cause,
        ra.is_suspended,
        ra.is_resumed,
        ra.has_furikae_yuso,
        ra.severity_score
    FROM bus b
    JOIN route_rail_links l ON l.route_id = b.route_id
    JOIN rail_alerts ra
      ON ra.line_id = l.line_id
     AND ra.observed_epoch >= b.bucket_15m_epoch - 2700
     AND ra.observed_epoch <  b.bucket_15m_epoch + 900
),

rail_agg AS (
    SELECT
        route_id,
        bucket_15m_epoch,
        -- (a) Structured status header only: no reading of the free text.
        MAX(status_category <> 'normal')                            AS status_any_abnormal,
        MAX(status_category = 'suspended')                          AS status_any_suspended,
        MAX(status_category = 'delay')                              AS status_any_delay,
        MAX(status_category = 'resumed')                            AS status_any_resumed,
        MAX(strong_link AND status_category <> 'normal')            AS status_strong_abnormal,
        -- (b) What the parser read from the bulletin text.
        MAX(is_disrupted)                                           AS rail_any_disrupted,
        MAX(is_suspended)                                           AS rail_any_suspended,
        MAX(is_resumed)                                             AS rail_any_resumed,
        MAX(has_furikae_yuso)                                       AS rail_any_furikae,
        MAX(severity_score)                                         AS rail_max_severity,
        MAX(CASE WHEN strong_link THEN severity_score ELSE 0 END)   AS rail_strong_max_severity,
        MAX(strong_link AND has_furikae_yuso)                       AS rail_strong_furikae,
        SUM(is_disrupted)                                           AS rail_disrupted_snapshots,
        -- (c) Raw text, for the TF-IDF embedding features.
        GROUP_CONCAT(DISTINCT CASE WHEN is_disrupted = 1 THEN text_ja END) AS recent_bulletins_ja
    FROM rail_recent
    GROUP BY route_id, bucket_15m_epoch
),

-- The cause behind the most severe recent bulletin (the latest one if tied).
top_cause AS (
    SELECT route_id, bucket_15m_epoch, parsed_cause AS top_rail_cause
    FROM (
        SELECT
            route_id,
            bucket_15m_epoch,
            parsed_cause,
            ROW_NUMBER() OVER (
                PARTITION BY route_id, bucket_15m_epoch
                ORDER BY severity_score DESC, observed_epoch DESC
            ) AS rank_in_bucket
        FROM rail_recent
        WHERE is_disrupted = 1
    )
    WHERE rank_in_bucket = 1
)

SELECT
    b.route_id,
    b.bucket_15m_epoch,
    b.n_vehicles,
    b.mean_delay_sec,
    b.p90_delay_sec,
    b.rolling_60m_delay_sec,
    b.lag_15m_delay_sec,
    b.lag_30m_delay_sec,
    b.mean_delay_sec - b.lag_15m_delay_sec AS delay_change_15m_sec,
    w.temp_c,
    w.precip_mm,
    w.wind_speed_ms,
    COALESCE(r.status_any_abnormal, 0)      AS status_any_abnormal,
    COALESCE(r.status_any_suspended, 0)     AS status_any_suspended,
    COALESCE(r.status_any_delay, 0)         AS status_any_delay,
    COALESCE(r.status_any_resumed, 0)       AS status_any_resumed,
    COALESCE(r.status_strong_abnormal, 0)   AS status_strong_abnormal,
    COALESCE(r.rail_any_disrupted, 0)       AS rail_any_disrupted,
    COALESCE(r.rail_any_suspended, 0)       AS rail_any_suspended,
    COALESCE(r.rail_any_resumed, 0)         AS rail_any_resumed,
    COALESCE(r.rail_any_furikae, 0)         AS rail_any_furikae,
    COALESCE(r.rail_max_severity, 0)        AS rail_max_severity,
    COALESCE(r.rail_strong_max_severity, 0) AS rail_strong_max_severity,
    COALESCE(r.rail_strong_furikae, 0)      AS rail_strong_furikae,
    COALESCE(r.rail_disrupted_snapshots, 0) AS rail_disrupted_snapshots,
    COALESCE(tc.top_rail_cause, 'none')     AS top_rail_cause,
    COALESCE(r.recent_bulletins_ja, '')     AS recent_bulletins_ja,
    -- Targets are NULL for the newest buckets (their future hasn't happened yet); training
    -- drops those rows, the dashboard forecasts them.
    b.target_delay_30m_sec,
    CASE WHEN b.target_delay_30m_sec IS NULL THEN NULL
         WHEN b.target_delay_30m_sec - b.rolling_60m_delay_sec >= 180
           OR b.target_delay_30m_sec >= 420
         THEN 1 ELSE 0 END                  AS target_surge_30m
FROM bus b
LEFT JOIN rail_agg r
       ON r.route_id = b.route_id AND r.bucket_15m_epoch = b.bucket_15m_epoch
LEFT JOIN top_cause tc
       ON tc.route_id = b.route_id AND tc.bucket_15m_epoch = b.bucket_15m_epoch
-- Weather from the last hour that was COMPLETE when the forecast is made (end of bucket).
-- Joining the bucket's own hour would use readings from later in that hour.
LEFT JOIN weather_hourly w
       ON w.hour_epoch = (b.bucket_15m_epoch + 900) - (b.bucket_15m_epoch + 900) % 3600 - 3600
ORDER BY b.bucket_15m_epoch, b.route_id;
