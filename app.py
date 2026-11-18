from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, BUS_ROUTES, RAIL_LINES, SQL_DIR
from src.ingest import generate_simulated_history
from src.nlp_parser import parse_japanese_bulletin
from src.train import load_sql_feature_table, prepare_design_matrix, run_training_and_ablation

st.set_page_config(
    page_title="Tokyo Multimodal Transit Cascade Predictor",
    page_icon="🚆",
    layout="wide",
)

st.title("🚆 Tokyo Multimodal Transit Disruption & Cascade Predictor")
st.caption(
    "Real-Time Cross-Modal Delay Forecasting: Fusing ODPT Japanese Rail Bulletins & Toei Bus GTFS-Realtime Telemetry"
)


@st.cache_resource
def ensure_pipeline_ready():
    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    if not bundle_path.exists():
        generate_simulated_history(days=14)
        run_training_and_ablation(artifacts_dir=ARTIFACTS_DIR)
    bundle = joblib.load(bundle_path)
    df = load_sql_feature_table()
    return bundle, df


bundle, df_features = ensure_pipeline_ready()
summary = bundle["summary"]

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📍 Live Corridor Monitor",
        "🈯 Japanese NLP Bulletin Inspector",
        "📊 Ablation & Model Validation",
        "🏗️ Architecture & SQL Lineage",
    ]
)

with tab1:
    st.subheader("Corridor Spillover Risk (30-Minute Horizon)")
    latest_bucket = int(df_features["bucket_15m_epoch"].max())
    latest_df = df_features[df_features["bucket_15m_epoch"] == latest_bucket].copy()

    X_latest, _, _ = prepare_design_matrix(latest_df, embedder=bundle["embedder"], fit_embedder=False)
    X_latest = X_latest.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    latest_df["pred_delay_30m_sec"] = bundle["model"].predict(X_latest)
    latest_df["pred_excess_sec"] = latest_df["pred_delay_30m_sec"] - latest_df["rolling_60m_delay_sec"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored Corridors", f"{len(latest_df)} Toei Routes")
    c2.metric("Active Hub Rail Alerts", int(latest_df["hub_rail_disrupted"].sum()))
    c3.metric("Max Predicted 30m Delay", f"{latest_df['pred_delay_30m_sec'].max():.0f} sec")
    c4.metric(
        "Multimodal RMSE Lift",
        f"-{summary['disruption_rmse_reduction_vs_tier2_pct']:.1f}%",
        help="Reduction in disruption-window RMSE after adding Japanese NLP rail alert features",
    )

    rail_lookup = {rl.line_id: rl for rl in RAIL_LINES}
    route_lookup = {br.route_id: br for br in BUS_ROUTES}

    display_rows = []
    map_points = []
    for _, row in latest_df.iterrows():
        rl = rail_lookup[row["nearest_rail_line_id"]]
        br = route_lookup[row["route_id"]]
        surge_risk = (
            "HIGH SURGE"
            if (row["pred_excess_sec"] >= 180 or row["pred_delay_30m_sec"] >= 420)
            else ("ELEVATED" if row["pred_excess_sec"] >= 90 else "NORMAL")
        )
        display_rows.append(
            {
                "Bus Route": f"{br.route_code} ({br.route_name_en})",
                "Anchor Hub": row["anchor_hub"],
                "Connected Rail Line": f"{rl.line_name_en} ({rl.line_name_ja})",
                "Rail Status": "DISRUPTED" if row["hub_rail_disrupted"] == 1 else "Normal (平常運転)",
                "Furikae (振替輸送)": "Active" if row["hub_furikae_active"] == 1 else "—",
                "Current Delay (s)": round(float(row["mean_delay_sec"]), 1),
                "Predicted +30m Delay (s)": round(float(row["pred_delay_30m_sec"]), 1),
                "Alert Level": surge_risk,
            }
        )
        map_points.append(
            {
                "lat": rl.hub_lat,
                "lon": rl.hub_lon,
                "hub": rl.primary_hub,
                "pred_delay": float(row["pred_delay_30m_sec"]),
            }
        )

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Tokyo Transfer Hub Telemetry Map")
    st.map(pd.DataFrame(map_points), latitude="lat", longitude="lon", size="pred_delay")

with tab2:
    st.subheader("Bilingual Railway Bulletin Parser & Spillover Simulator")
    st.write(
        "Test how raw Japanese `odpt:TrainInformation` bulletins are parsed into structured operational severity and cross-modal spillover signals."
    )

    presets = {
        "Odakyu Line Track Accident + Furikae Transfer (Shinjuku)": (
            "運転見合わせ",
            "【小田急小田原線 運転見合わせ】新宿駅構内での人身事故の影響で、上下線で運転を見合わせています。現在、都営バスおよび周辺路線にて振替輸送を実施しています。",
        ),
        "Tokyu Den-en-toshi Rolling Stock Fault (Shibuya)": (
            "遅延",
            "【東急田園都市線 遅延】渋谷駅での車両点検の影響で、上下線の一部列車に25分程度の遅れと運休が出ています。振替輸送をご利用ください。",
        ),
        "JR Chuo Rapid High Wind Speed Restriction": (
            "ダイヤ乱れ",
            "【JR中央線快速 ダイヤ乱れ】強風による速度規制および信号確認の影響で、直通運転を中止しダイヤが大幅に乱れています。",
        ),
        "Normal Service (平常運転)": (
            "平常運転",
            "【JR山手線】現在、平常どおり運転しています。",
        ),
    }

    selected_preset = st.selectbox("Load Sample ODPT Bulletin", list(presets.keys()))
    default_status, default_text = presets[selected_preset]

    status_input = st.text_input("Status Header (`odpt:trainInformationStatus`)", value=default_status)
    text_input = st.text_area("Bulletin Body (`odpt:trainInformationText`)", value=default_text, height=110)

    parsed = parse_japanese_bulletin(text_input, status_input)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Disruption Detected", "YES" if parsed.is_disrupted else "NO")
    p2.metric("Operational Severity", f"{parsed.severity_score:.2f} / 1.00")
    p3.metric("Furikae Transfer (振替輸送)", "ACTIVE" if parsed.has_furikae_yuso else "Inactive")
    p4.metric("Full Suspension", "YES" if parsed.is_suspended else "No")

    st.info(f"**Classified Incident Cause:** {parsed.cause_en}")
    st.write("**Extracted Operational Keywords:**", ", ".join(parsed.matched_keywords) or "None")

with tab3:
    st.subheader("Chronological Holdout Ablation Study (80% Train / 20% Future Test)")
    st.write(
        "Evaluating whether unstructured Japanese railway bulletins improve 30-minute downstream bus delay forecasting beyond timetable baselines and SQL rolling telemetry."
    )

    tiers_df = pd.DataFrame(summary["tiers"])
    st.dataframe(tiers_df, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    ablation_img = ARTIFACTS_DIR / "ablation_comparison.png"
    fi_img = ARTIFACTS_DIR / "feature_importance.png"
    if ablation_img.exists():
        col_left.image(str(ablation_img), caption="RMSE Comparison Across Feature Tiers")
    if fi_img.exists():
        col_right.image(str(fi_img), caption="Top Multimodal Predictors (XGBoost Gain)")

with tab4:
    st.subheader("SQL Feature Engineering Lineage & Governance")
    st.markdown(
        """
        - **Relational Store:** Normalized 5-table SQLite/PostgreSQL schema (`rail_lines`, `bus_routes`, `rail_alerts`, `bus_telemetry`, `weather_hourly`).
        - **Windowed SQL Pipeline (`sql/features.sql`):** Uses CTEs, `AVG() OVER (ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)`, `LAG()`, and `LEAD()` strictly partitioned by route and ordered by 15-minute epoch buckets to eliminate lookahead leakage.
        - **Design Artifacts:** Full Mermaid system diagram, Entity-Relationship Diagram (ERD), and Model Card are maintained in `README.md` and `docs/MODEL_CARD.md`.
        """
    )
    with st.expander("Inspect `sql/features.sql`"):
        st.code((SQL_DIR / "features.sql").read_text(encoding="utf-8"), language="sql")
