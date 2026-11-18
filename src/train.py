"""Train and evaluate 30-minute bus-delay forecasts, adding one information source at a time.

    python -m src.train

Models A-F each add one kind of information to the one before, so the comparison shows
what each source is worth. On synthetic data this is a pipeline check (does it recover the
effect the simulator planted?); on collected data it is the real experiment.

Evaluation is "rolling origin": the newest half of the timeline is cut into 4 successive
test blocks, and each block is forecast by models trained only on data from before it.
One unusual incident therefore can't decide the result on its own.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, root_mean_squared_error

from src import charts
from src.config import ARTIFACTS_DIR, BUS_ROUTES, HORIZON_SECONDS, SQL_DIR, get_connection
from src.ingest import read_metadata
from src.nlp_parser import CAUSE_RULES, UNKNOWN_CAUSE, BulletinTextEmbedder
from src.simulate import JST

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET = "target_delay_30m_sec"
TIME_COLS = ["hour_sin", "hour_cos", "day_of_week", "is_weekend"]
BUS_WEATHER_COLS = ["n_vehicles", "mean_delay_sec", "p90_delay_sec", "rolling_60m_delay_sec", "lag_15m_delay_sec",
                    "lag_30m_delay_sec", "delay_change_15m_sec", "temp_c", "precip_mm", "wind_speed_ms"]
STATUS_COLS = ["status_any_abnormal", "status_any_suspended", "status_any_delay", "status_any_resumed",
               "status_strong_abnormal"]
PARSED_COLS = ["rail_any_disrupted", "rail_any_suspended", "rail_any_resumed", "rail_any_furikae",
               "rail_max_severity", "rail_strong_max_severity", "rail_strong_furikae", "rail_disrupted_snapshots"]
ROUTE_LEVELS = [r.route_id for r in BUS_ROUTES]
CAUSE_LEVELS = [c[0] for c in CAUSE_RULES] + [UNKNOWN_CAUSE[0], "none"]

PERSISTENCE = "A. Persistence (delay in 30 min = delay now)"
MODEL_NAMES = [
    PERSISTENCE,
    "B. Time of day + route",
    "C. + recent bus delays & weather",
    "D. + rail status header (no text)",
    "E. + bulletin text, parsed by rules",
    "F. + bulletin text, TF-IDF embeddings",
]
N_FOLDS = 4
MIN_LABELED_BUCKETS = 200  # about 3 days of bus service
SURGE_ABOVE_RECENT_SEC, SURGE_ABSOLUTE_SEC = 180.0, 420.0


def load_feature_table(db_path: str | Path | None = None) -> pd.DataFrame:
    """Run sql/features.sql and add calendar features. Includes the newest rows, whose
    target is still unknown (NULL)."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query((SQL_DIR / "features.sql").read_text(encoding="utf-8"), conn)
    finally:
        conn.close()
    if df.empty:
        raise SystemExit("The feature query returned no rows. Load data first: `python -m src.ingest simulate`.")
    local = pd.to_datetime(df["bucket_15m_epoch"], unit="s", utc=True).dt.tz_convert(JST)
    hours = local.dt.hour + local.dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["day_of_week"] = local.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def rolling_origin_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, first_test_at: float = 0.5
                         ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Successive (train, test) splits in time order -- never shuffled.

    Each test block covers the next slice of the timeline; its model trains on everything
    before it, minus the last 30 minutes: those rows' *targets* fall inside the test block,
    so keeping them would leak a little of the answer into training.
    """
    buckets = np.sort(df["bucket_15m_epoch"].unique())
    edges = [int(len(buckets) * f) for f in np.linspace(first_test_at, 1.0, n_folds + 1)]
    for start, stop in zip(edges[:-1], edges[1:]):
        t0, t1 = buckets[start], buckets[stop - 1]
        train = df[df["bucket_15m_epoch"] < t0 - HORIZON_SECONDS].reset_index(drop=True)
        test = df[df["bucket_15m_epoch"].between(t0, t1)].reset_index(drop=True)
        yield train, test


def make_features(df: pd.DataFrame, embedder: BulletinTextEmbedder, fit: bool):
    """All model inputs plus the column list for each model. Fixed category levels keep
    train/test/dashboard columns identical."""
    routes = pd.get_dummies(pd.Categorical(df["route_id"], categories=ROUTE_LEVELS), prefix="route", dtype=float)
    causes = pd.get_dummies(pd.Categorical(df["top_rail_cause"], categories=CAUSE_LEVELS), prefix="cause", dtype=float)
    texts = df["recent_bulletins_ja"].fillna("").tolist()
    emb = embedder.fit_transform(texts) if fit else embedder.transform(texts)
    emb = pd.DataFrame(emb, columns=[f"text_emb_{i}" for i in range(emb.shape[1])])
    X = pd.concat([df[TIME_COLS + BUS_WEATHER_COLS + STATUS_COLS + PARSED_COLS].reset_index(drop=True),
                   routes, causes, emb], axis=1).astype(float)
    b = TIME_COLS + list(routes.columns)
    c = b + BUS_WEATHER_COLS
    d = c + STATUS_COLS
    e = d + PARSED_COLS + list(causes.columns)
    f = e + list(emb.columns)
    return X, dict(zip(MODEL_NAMES[1:], [b, c, d, e, f]))


def build_model(seed: int = 42):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.85,
                            colsample_bytree=0.85, reg_lambda=1.5, random_state=seed, n_jobs=-1)
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=250, max_depth=5, learning_rate=0.05, random_state=seed)


def fit_predict(X_train: pd.DataFrame, train: pd.DataFrame, X_test: pd.DataFrame, test: pd.DataFrame,
                cols: list[str], predict_change: bool = True):
    """Fit one model and forecast the test rows.

    With predict_change, the model learns the *change* from the current delay
    (target - mean_delay_sec) and the forecast is current delay + predicted change. Tree
    models can't output values outside the range they saw in training, so anchoring on
    "now" lets a forecast follow an unusually large disruption instead of capping it.
    """
    model = build_model()
    if predict_change:
        model.fit(X_train[cols], train[TARGET] - train["mean_delay_sec"])
        return model, test["mean_delay_sec"].to_numpy() + model.predict(X_test[cols])
    model.fit(X_train[cols], train[TARGET])
    return model, model.predict(X_test[cols])


def _scores(y: np.ndarray, pred: np.ndarray, test: pd.DataFrame, disrupted: np.ndarray) -> dict[str, Any]:
    recent = test["rolling_60m_delay_sec"].to_numpy()
    pred_surge = ((pred - recent >= SURGE_ABOVE_RECENT_SEC) | (pred >= SURGE_ABSOLUTE_SEC)).astype(int)
    rmse = lambda a, b: round(float(root_mean_squared_error(a, b)), 2) if len(a) else None  # noqa: E731
    mae = lambda a, b: round(float(mean_absolute_error(a, b)), 2) if len(a) else None  # noqa: E731
    return {
        "rmse_sec": rmse(y, pred),
        "mae_sec": mae(y, pred),
        "disruption_rmse_sec": rmse(y[disrupted], pred[disrupted]),
        "disruption_mae_sec": mae(y[disrupted], pred[disrupted]),
        "surge_f1": round(float(f1_score(test["target_surge_30m"], pred_surge, zero_division=0)), 3),
    }


def evaluate(db_path: str | Path | None = None, df: pd.DataFrame | None = None,
             n_folds: int = N_FOLDS, predict_change: bool = True) -> dict[str, Any]:
    df = load_feature_table(db_path) if df is None else df
    labeled = df[df[TARGET].notna()].reset_index(drop=True)
    if labeled["bucket_15m_epoch"].nunique() < MIN_LABELED_BUCKETS:
        raise SystemExit(f"Only {labeled['bucket_15m_epoch'].nunique()} usable 15-minute intervals so far "
                         f"(need {MIN_LABELED_BUCKETS}+, i.e. a few days). Keep collecting, then retry.")
    preds = {name: [] for name in MODEL_NAMES}
    tests, fold_rmse = [], {name: [] for name in MODEL_NAMES}
    for train, test in rolling_origin_folds(labeled, n_folds):
        embedder = BulletinTextEmbedder(n_components=8)  # refit per fold: training text only
        X_train, tiers = make_features(train, embedder, fit=True)
        X_test, _ = make_features(test, embedder, fit=False)
        fold_preds = {PERSISTENCE: test["mean_delay_sec"].to_numpy()}
        for name, cols in tiers.items():
            _, fold_preds[name] = fit_predict(X_train, train, X_test, test, cols, predict_change)
        for name, p in fold_preds.items():
            preds[name].append(p)
            fold_rmse[name].append(round(float(root_mean_squared_error(test[TARGET], p)), 2))
        tests.append(test)

    test = pd.concat(tests, ignore_index=True)
    y = test[TARGET].to_numpy()
    # "Disruption window": a linked rail line showed trouble (header or text) in the last hour.
    disrupted = ((test["rail_any_disrupted"] == 1) | (test["status_any_abnormal"] == 1)).to_numpy()
    models = []
    for name in MODEL_NAMES:
        models.append({"model": name, **_scores(y, np.concatenate(preds[name]), test, disrupted),
                       "rmse_by_test_block": fold_rmse[name]})
        log.info("%-44s RMSE %7.2f | disruption RMSE %s", name, models[-1]["rmse_sec"], models[-1]["disruption_rmse_sec"])

    first, last = (pd.Timestamp(int(t), unit="s", tz="UTC").tz_convert(JST).isoformat()
                   for t in (test["bucket_15m_epoch"].min(), test["bucket_15m_epoch"].max()))
    return {
        "data": read_metadata(db_path),
        "evaluation": f"rolling origin, {n_folds} successive test blocks from {first[:16]} to {last[:16]} (JST)",
        "models_predict": "change from current delay" if predict_change else "delay level",
        "labeled_rows": len(labeled),
        "test_rows": len(test),
        "test_rows_in_disruption_windows": int(disrupted.sum()),
        "test_surge_events": int(test["target_surge_30m"].sum()),
        "models": models,
    }


def fit_final_model(df: pd.DataFrame) -> dict[str, Any]:
    """After evaluating, refit the full model (F) on every labelled row, for the dashboard."""
    labeled = df[df[TARGET].notna()].reset_index(drop=True)
    embedder = BulletinTextEmbedder(n_components=8)
    X, tiers = make_features(labeled, embedder, fit=True)
    cols = tiers[MODEL_NAMES[-1]]
    model = build_model()
    model.fit(X[cols], labeled[TARGET] - labeled["mean_delay_sec"])
    return {"model": model, "feature_columns": cols, "embedder": embedder, "predicts_change": True}


def forecast(bundle: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    """Forecast 30-minute-ahead delay for feature rows using a saved model bundle."""
    X, _ = make_features(rows, bundle["embedder"], fit=False)
    change = bundle["model"].predict(X[bundle["feature_columns"]])
    return rows["mean_delay_sec"].to_numpy() + change


def run(db_path: str | Path | None = None, artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = load_feature_table(db_path)
    report = evaluate(db_path, df=df)
    (artifacts_dir / "ablation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report["models"]).drop(columns="rmse_by_test_block").to_csv(artifacts_dir / "metrics_summary.csv", index=False)

    bundle = {**fit_final_model(df), "report": report}
    joblib.dump(bundle, artifacts_dir / "model_bundle.joblib")
    _plot_ablation(report, artifacts_dir / "ablation_comparison.png")
    _plot_importance(bundle["model"], bundle["feature_columns"], report, artifacts_dir / "feature_importance.png")
    return report


def data_label(meta: dict[str, str]) -> str:
    if meta.get("data_source") == "simulated":
        effect = "planted rail->bus effect" if meta.get("sim_planted_effect") == "True" else "PLACEBO: no rail->bus effect"
        return f"Synthetic data ({effect}, seed {meta.get('sim_seed')}) - a pipeline check, not a real-world result"
    if meta.get("data_source") == "collected":
        return f"Real ODPT/Toei data collected {meta.get('first_snapshot', '?')[:10]} to {meta.get('last_snapshot', '?')[:10]}"
    return "Data source unknown"


def _plot_ablation(report: dict[str, Any], path: Path) -> None:
    models = report["models"]
    labels = [m["model"] for m in models]
    colors = [charts.MUTED if m["model"] == PERSISTENCE else charts.SERIES[0] for m in models]
    fig, (left, right) = charts.figure(n_panels=2, width=9.0, height=3.5)
    fig.subplots_adjust(top=0.78, wspace=0.08)
    charts.hbars(left, labels, [m["rmse_sec"] for m in models], colors, panel_title="All test rows")
    charts.hbars(right, [""] * len(labels), [m["disruption_rmse_sec"] or 0 for m in models], colors,
                 panel_title=f"Rail-disruption windows ({report['test_rows_in_disruption_windows']:,} rows)")
    right.set_yticks([])
    charts.titles(fig, "30-minute bus delay forecast error, RMSE in seconds (lower is better)",
                  data_label(report["data"]) + f"; {N_FOLDS} rolling test blocks")
    charts.save(fig, path)


def _plot_importance(model, cols: list[str], report: dict[str, Any], path: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(12)
    fig, (ax,) = charts.figure(width=7.0, height=3.6)
    fig.subplots_adjust(top=0.8)
    charts.hbars(ax, list(imp.index), list(imp.values), [charts.SERIES[0]] * len(imp), fmt="{:.2f}")
    charts.titles(fig, "What the full model (F) relies on: top 12 features by XGBoost gain", data_label(report["data"]))
    charts.save(fig, path)


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=2, ensure_ascii=False))
    print(pd.DataFrame(out["models"]).drop(columns="rmse_by_test_block").to_string(index=False))
