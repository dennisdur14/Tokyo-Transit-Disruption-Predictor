"""Dashboard for the pipeline:  streamlit run app.py

Shows whatever is in the database -- synthetic data until you load collected snapshots
(`python -m src.ingest snapshots && python -m src.train`). It is not a live feed.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, BUS_ROUTES, RAIL_LINES, ROUTE_RAIL_LINKS, SQL_DIR
from src.ingest import load_simulation, read_metadata
from src.nlp_parser import parse_bulletin, status_category
from src.simulate import JST
from src.train import data_label, forecast, load_feature_table
from src.train import run as run_training

st.set_page_config(page_title="Tokyo Rail-to-Bus Spillover", page_icon="🚌", layout="wide")


@st.cache_resource
def load_pipeline():
    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    if not bundle_path.exists():  # first launch: build everything from the default simulation
        if not read_metadata():
            load_simulation()
        run_training()
    return joblib.load(bundle_path), load_feature_table(), read_metadata()


bundle, features, meta = load_pipeline()
report = bundle["report"]

st.title("🚌 Tokyo rail-to-bus spillover")
st.caption("Do Tokyo rail disruptions make nearby Toei buses run later, and can the rail bulletins help forecast it?")
if meta.get("data_source") == "simulated":
    st.warning(f"**{data_label(meta)}.** Numbers here show the pipeline works on data with a known answer; "
               "they say nothing about real Tokyo buses.", icon="⚠️")
else:
    st.info(data_label(meta), icon="📡")

monitor, parser_tab, models_tab, how_tab = st.tabs(
    ["Corridor monitor", "Bulletin parser", "Model comparison", "How it works"])

with monitor:
    latest_epoch = int(features["bucket_15m_epoch"].max())
    latest = features[features["bucket_15m_epoch"] == latest_epoch].copy()
    latest["forecast"] = forecast(bundle, latest)
    st.subheader("Forecast for the bucket starting 30 minutes after the latest interval")
    st.caption(f"Latest 15-minute interval in the database: "
               f"{pd.Timestamp(latest_epoch, unit='s', tz='UTC').tz_convert(JST):%Y-%m-%d %H:%M} JST")

    routes = {r.route_id: r for r in BUS_ROUTES}
    lines = {l.line_id: l for l in RAIL_LINES}
    linked = {}
    for route_id, line_id, relation in ROUTE_RAIL_LINKS:
        linked.setdefault(route_id, []).append(f"{lines[line_id].name_en} ({relation.replace('_', ' ')})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Bus routes", len(latest))
    c2.metric("Routes with a linked rail disruption", int(latest["rail_any_disrupted"].sum()))
    c3.metric("Largest forecast delay", f"{latest['forecast'].max() / 60:.1f} min")

    table = pd.DataFrame({
        "Route": [f"{routes[r].route_code}  {routes[r].name_en}" for r in latest["route_id"]],
        "Delay now (min)": (latest["mean_delay_sec"] / 60).round(1).to_numpy(),
        "Forecast (min)": (latest["forecast"] / 60).round(1).to_numpy(),
        "Linked rail": ["DISRUPTED" if d else "normal" for d in latest["rail_any_disrupted"]],
        "振替輸送": ["yes" if f else "" for f in latest["rail_any_furikae"]],
        "Linked lines": ["; ".join(linked.get(r, [])) for r in latest["route_id"]],
    })
    st.dataframe(table, width="stretch", hide_index=True)
    st.map(pd.DataFrame({
        "lat": [routes[r].lat for r in latest["route_id"]],
        "lon": [routes[r].lon for r in latest["route_id"]],
        "size": (latest["forecast"].clip(lower=30) * 2).to_numpy(),
    }), latitude="lat", longitude="lon", size="size")
    st.caption("Dots sit at each route's centre; size grows with the forecast delay.")

with parser_tab:
    st.subheader("Try the Japanese bulletin parser")
    st.write("Rules, not a learned model: every output traces back to the words that triggered it. "
             "Its accuracy on real bulletins is measured against hand labels (`python -m src.parser_eval score`).")
    presets = {
        "Real Toei text: normal service": ("", "現在、１５分以上の遅延はありません。"),
        "Suspension + 振替輸送": ("運転見合わせ", "新宿駅構内での人身事故の影響で、上下線で運転を見合わせています。"
                                          "現在、都営バスで振替輸送を行っています。"),
        "Resumed after a suspension": ("", "人身事故の影響で運転を見合わせていましたが、10時15分頃に運転を再開しました。現在も遅れが出ています。"),
        "Delay, 振替輸送 NOT offered": ("遅延", "車両点検の影響で遅れが出ています。なお、振替輸送は実施しておりません。"),
        "Station name containing 雪": ("", "雪が谷大塚駅での急病人救護の影響で、一部列車に遅れが出ています。"),
    }
    header, text = presets[st.selectbox("Example", list(presets))]
    header = st.text_input("Status header (odpt:trainInformationStatus; often empty)", value=header)
    text = st.text_area("Bulletin text (odpt:trainInformationText)", value=text, height=100)
    p = parse_bulletin(text, header)
    a, b, c, d = st.columns(4)
    a.metric("Disrupted", "yes" if p.is_disrupted else "no")
    b.metric("Suspended now", "yes" if p.is_suspended else "no")
    c.metric("Resumed", "yes" if p.is_resumed else "no")
    d.metric("振替輸送 in effect", "yes" if p.has_furikae_yuso else "no")
    st.write(f"**Cause:** {p.cause_en}  ·  **Severity (hand-set heuristic):** {p.severity_score:.2f}  ·  "
             f"**Header category alone:** {status_category(header)}")
    st.write("**Matched:** " + (", ".join(p.matched_keywords) or "nothing"))

with models_tab:
    st.subheader("What each information source adds")
    st.write(f"Evaluation: {report['evaluation']}. Every model is trained only on data from before the block "
             "it forecasts. A = no model at all; B-F each add one source to the model before.")
    cols = ["model", "rmse_sec", "mae_sec", "disruption_rmse_sec", "disruption_mae_sec", "surge_f1"]
    st.dataframe(pd.DataFrame(report["models"])[cols], width="stretch", hide_index=True)
    st.caption(f"{report['test_rows']:,} test rows, {report['test_rows_in_disruption_windows']:,} in rail-disruption "
               f"windows, {report['test_surge_events']} surge events (≥3 min above the past hour, or ≥7 min late).")
    for name in ("ablation_comparison.png", "sanity_check.png", "feature_importance.png"):
        if (ARTIFACTS_DIR / name).exists():
            st.image(str(ARTIFACTS_DIR / name), width="stretch")
    sanity = ARTIFACTS_DIR / "sanity_check.json"
    if sanity.exists():
        with st.expander("Placebo check numbers"):
            st.json(json.loads(sanity.read_text(encoding="utf-8")))

with how_tab:
    st.subheader("Pipeline")
    st.markdown("""
1. **Collect** (`src/collect.py`, every 15 min via GitHub Actions): Toei bus positions with a delay estimate
   per bus (from the published timetable), ODPT rail bulletins, Open-Meteo weather.
2. **Load** (`src/ingest.py`): snapshots (or a simulation) into SQLite; the parser reads every bulletin.
3. **Features** (`sql/features.sql`): one row per route per 15 minutes -- rolling delays, gap-safe lags, and
   bulletins from the route's linked rail lines in the last hour.
4. **Evaluate** (`src/train.py`): six models, rolling-origin test blocks, persistence baseline.
5. **Check** (`src/sanity_check.py`): the same evaluation on simulated worlds with and without a planted effect.
""")
    with st.expander("sql/features.sql"):
        st.code((SQL_DIR / "features.sql").read_text(encoding="utf-8"), language="sql")
