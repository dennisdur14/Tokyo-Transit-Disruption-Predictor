from __future__ import annotations

import json
import logging
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, root_mean_squared_error

from src.config import ARTIFACTS_DIR, SQL_DIR, get_connection
from src.nlp_parser import BilingualBulletinEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


def _build_regressor(random_state: int = 42):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=260,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            random_state=random_state,
            n_jobs=2,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=260,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
        )


def load_sql_feature_table(db_url: str | None = None) -> pd.DataFrame:
    query = (SQL_DIR / "features.sql").read_text(encoding="utf-8")
    conn = get_connection(db_url)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("Feature query returned 0 rows. Run `python -m src.ingest --mode simulate` first.")

    dt_jst = pd.to_datetime(df["bucket_15m_epoch"], unit="s", utc=True).dt.tz_convert(JST)
    df["hour_jst"] = dt_jst.dt.hour
    df["minute_jst"] = dt_jst.dt.minute
    df["day_of_week"] = dt_jst.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["is_weekend"] == 0) & (df["hour_jst"].between(7, 9) | df["hour_jst"].between(17, 20)))
    ).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour_jst"] + df["minute_jst"] / 60.0) / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour_jst"] + df["minute_jst"] / 60.0) / 24.0)
    return df


def prepare_design_matrix(
    df: pd.DataFrame,
    embedder: BilingualBulletinEmbedder | None = None,
    fit_embedder: bool = True,
) -> tuple[pd.DataFrame, BilingualBulletinEmbedder, dict[str, list[str]]]:
    route_dummies = pd.get_dummies(df["route_id"], prefix="route", dtype=float)
    cause_dummies = pd.get_dummies(df["dominant_rail_cause"], prefix="cause", dtype=float)

    tier1_cols = [
        "hour_jst",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
        "hour_sin",
        "hour_cos",
        "hub_distance_km",
    ]

    tier2_extra_cols = [
        "active_vehicles",
        "mean_delay_sec",
        "p90_delay_sec",
        "mean_headway_sec",
        "headway_drift_sec",
        "occupancy_ratio",
        "rolling_60m_delay_sec",
        "rolling_60m_headway_sec",
        "lag_15m_delay_sec",
        "lag_30m_delay_sec",
        "delay_velocity_15m",
        "temp_c",
        "precip_mm",
        "wind_speed_ms",
    ]

    tier3_structured_nlp_cols = [
        "hub_rail_disrupted",
        "hub_max_rail_severity",
        "hub_furikae_active",
        "hub_active_alert_count",
    ]

    if embedder is None:
        embedder = BilingualBulletinEmbedder(n_components=8, use_neural=False)

    if fit_embedder:
        emb_mat = embedder.fit_transform(df["active_alerts_ja"].fillna("").tolist())
    else:
        emb_mat = embedder.transform(df["active_alerts_ja"].fillna("").tolist())

    emb_cols = [f"ja_emb_{i}" for i in range(emb_mat.shape[1])]
    emb_df = pd.DataFrame(emb_mat, columns=emb_cols, index=df.index)

    X_full = pd.concat(
        [
            df[tier1_cols + tier2_extra_cols + tier3_structured_nlp_cols],
            route_dummies,
            cause_dummies,
            emb_df,
        ],
        axis=1,
    )

    feature_tiers = {
        "Tier 1: Timetable & Temporal Baseline": tier1_cols + list(route_dummies.columns),
        "Tier 2: + SQL Rolling Telemetry & Weather": tier1_cols + list(route_dummies.columns) + tier2_extra_cols,
        "Tier 3: + Japanese NLP Rail Alerts (Full Multimodal)": list(X_full.columns),
    }
    return X_full, embedder, feature_tiers


def run_training_and_ablation(db_url: str | None = None, artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = load_sql_feature_table(db_url)

    # Strict chronological split (first 80% train, final 20% holdout test — zero future leakage)
    unique_buckets = np.sort(df["bucket_15m_epoch"].unique())
    split_idx = int(len(unique_buckets) * 0.80)
    cutoff_epoch = int(unique_buckets[split_idx])

    train_mask = df["bucket_15m_epoch"] < cutoff_epoch
    test_mask = df["bucket_15m_epoch"] >= cutoff_epoch

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)

    X_train_full, embedder, feature_tiers = prepare_design_matrix(df_train, fit_embedder=True)
    X_test_full, _, _ = prepare_design_matrix(df_test, embedder=embedder, fit_embedder=False)
    X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0.0)

    y_train = df_train["target_delay_30m_sec"].to_numpy()
    y_test = df_test["target_delay_30m_sec"].to_numpy()
    y_test_surge = df_test["target_surge_30m"].to_numpy()
    rolling_base_test = df_test["rolling_60m_delay_sec"].to_numpy()

    ablation_results: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    for tier_name, cols in feature_tiers.items():
        valid_cols = [c for c in cols if c in X_train_full.columns]
        model = _build_regressor(random_state=42)
        model.fit(X_train_full[valid_cols], y_train)
        preds = model.predict(X_test_full[valid_cols])

        rmse = float(root_mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))

        pred_surge_score = np.maximum(preds - rolling_base_test, preds - 240.0)
        pred_surge_bin = ((preds - rolling_base_test >= 180.0) | (preds >= 420.0)).astype(int)
        surge_f1 = float(f1_score(y_test_surge, pred_surge_bin, zero_division=0))
        surge_pr_auc = float(average_precision_score(y_test_surge, pred_surge_score))

        disrupted_idx = df_test["hub_rail_disrupted"].to_numpy() == 1
        disrupted_rmse = (
            float(root_mean_squared_error(y_test[disrupted_idx], preds[disrupted_idx]))
            if disrupted_idx.any()
            else rmse
        )

        ablation_results.append(
            {
                "tier": tier_name,
                "num_features": len(valid_cols),
                "rmse_sec": round(rmse, 2),
                "mae_sec": round(mae, 2),
                "disruption_window_rmse_sec": round(disrupted_rmse, 2),
                "surge_f1": round(surge_f1, 4),
                "surge_pr_auc": round(surge_pr_auc, 4),
            }
        )
        models[tier_name] = (model, valid_cols)
        logger.info(
            "%s -> RMSE: %.2fs | Disruption RMSE: %.2fs | Surge F1: %.3f | PR-AUC: %.3f",
            tier_name,
            rmse,
            disrupted_rmse,
            surge_f1,
            surge_pr_auc,
        )

    t1_rmse = ablation_results[0]["rmse_sec"]
    t2_rmse = ablation_results[1]["rmse_sec"]
    t3_rmse = ablation_results[2]["rmse_sec"]
    t2_dis_rmse = ablation_results[1]["disruption_window_rmse_sec"]
    t3_dis_rmse = ablation_results[2]["disruption_window_rmse_sec"]

    summary = {
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "rmse_reduction_vs_tier1_pct": round((t1_rmse - t3_rmse) / t1_rmse * 100.0, 2),
        "rmse_reduction_vs_tier2_pct": round((t2_rmse - t3_rmse) / t2_rmse * 100.0, 2),
        "disruption_rmse_reduction_vs_tier2_pct": round((t2_dis_rmse - t3_dis_rmse) / t2_dis_rmse * 100.0, 2),
        "tiers": ablation_results,
    }

    (artifacts_dir / "ablation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(ablation_results).to_csv(artifacts_dir / "metrics_summary.csv", index=False)

    full_model, full_cols = models["Tier 3: + Japanese NLP Rail Alerts (Full Multimodal)"]
    joblib.dump(
        {
            "model": full_model,
            "feature_columns": full_cols,
            "embedder": embedder,
            "summary": summary,
        },
        artifacts_dir / "model_bundle.joblib",
    )

    _save_plots(ablation_results, full_model, full_cols, artifacts_dir)
    return summary


def _save_plots(
    ablation_results: list[dict[str, Any]],
    full_model: Any,
    full_cols: list[str],
    artifacts_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels = ["Tier 1\nTimetable Only", "Tier 2\n+ SQL Telemetry & Weather", "Tier 3\n+ Japanese NLP Rail Alerts"]
    overall_rmse = [r["rmse_sec"] for r in ablation_results]
    disrupt_rmse = [r["disruption_window_rmse_sec"] for r in ablation_results]

    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, overall_rmse, width, label="Overall Holdout RMSE (s)", color="#3b6998")
    b2 = ax.bar(x + width / 2, disrupt_rmse, width, label="Rail Disruption Window RMSE (s)", color="#d95f02")

    ax.set_ylabel("30-Min Forecast RMSE (seconds)")
    ax.set_title("Ablation Study: Value of Japanese Rail NLP Features on Bus Corridor Forecasting")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "ablation_comparison.png", dpi=160)
    plt.close(fig)

    if hasattr(full_model, "feature_importances_"):
        imp = pd.Series(full_model.feature_importances_, index=full_cols).sort_values(ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        imp.plot(kind="barh", ax=ax, color="#2b5c8f")
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title("Top 12 Predictors of 30-Minute Downstream Bus Corridor Delay")
        fig.tight_layout()
        fig.savefig(artifacts_dir / "feature_importance.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    report = run_training_and_ablation()
    print(json.dumps(report, indent=2))
