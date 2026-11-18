# Model Card: Tokyo Multimodal Transit Disruption & Cascade Predictor

## 1. Model Details
- **Architecture:** Gradient Boosted Decision Tree (`XGBRegressor`, max depth 5, 260 estimators, L2 regularization $\lambda = 1.5$) paired with a domain-calibrated Japanese railway bulletin parser and dense subword/multilingual text embeddings.
- **Forecast Horizon:** +30 minutes (`target_delay_30m_sec`) at 15-minute spatial-temporal corridor resolution across 8 major Toei Bus transfer corridors in Tokyo.

## 2. Intended Use & Users
- **Primary Users:** Transit operations dispatchers, municipal mobility planners, and emergency response coordinators.
- **Intended Use:** Anticipating surface bus overcrowding, headway bunching, and *振替輸送* (alternative transit transfer) spillover surges 30 minutes after upstream urban rail incidents.
- **Out-of-Scope Use:** Automated safety-critical rail signaling control or individual driver disciplinary evaluation.

## 3. Data Sources & Leakage Controls
- **Structured Sources:** Toei Bus GTFS-Realtime (`TripUpdate`), static corridor topology (`bus_routes`, `rail_lines`), and Open-Meteo Tokyo hourly weather (`temp_c`, `precip_mm`, `wind_speed_ms`).
- **Unstructured Source:** ODPT `odpt:TrainInformation` Japanese service-status bulletins across 8 Tokyo rail lines.
- **Temporal Leakage Prevention:**
  - All SQL window functions (`AVG() OVER (ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)` and `LAG()`) in `sql/features.sql` strictly look backward in time.
  - Rail alert joins restrict `ra.observed_epoch` to `[bucket_15m_epoch - 2700, bucket_15m_epoch]`.
  - Model evaluation enforces a strict **80% past / 20% future chronological split** rather than randomized cross-validation.

## 4. Surge Alert Definition
A corridor is flagged for a 30-minute surge (`target_surge_30m = 1`) when predicted delay exceeds the route's rolling 60-minute baseline by $\ge 180\text{ s}$ or reaches $\ge 420\text{ s}$ absolute delay.

## 5. Known Limitations & Failure Modes
1. **GTFS-Realtime Packet Drops:** During severe congestion or cellular outages, vehicle `TripUpdate` feeds may under-report active bus counts.
2. **Exogenous Non-Rail Events:** Stadium concerts, fireworks festivals, or street closures create localized road congestion not reflected in rail bulletins.
3. **Cold-Start Corridors:** Newly added bus routes without mapped transfer-hub distances default to zero rail spillover until registered in `bus_routes`.
