"""Placebo test: does the pipeline find a rail->bus effect only when one exists?

    python -m src.sanity_check          # 5 seeds x (planted, placebo), about a minute

For each seed it simulates the same 28 days twice: once with the planted rail->bus effect
and once with it switched off (identical weather, incidents, bulletins and bus noise). Both
go through the exact evaluation used on real data. Rail features should improve forecasts
in the planted world and do nothing in the placebo world -- if they "help" there, some
information is leaking.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from src import charts
from src.config import ARTIFACTS_DIR
from src.ingest import load_simulation
from src.simulate import SimConfig
from src.train import MODEL_NAMES, evaluate

log = logging.getLogger(__name__)
BASE_MODEL = MODEL_NAMES[2]  # C: bus delays + weather, no rail information
RAIL_MODELS = MODEL_NAMES[3:]  # D, E, F


def run(seeds=(1, 2, 3, 4, 5), days: int = 28, artifacts_dir: Path = ARTIFACTS_DIR,
        predict_change: bool = True) -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            for planted in (True, False):
                db = Path(tmp) / f"sim_{seed}_{planted}.db"
                load_simulation(SimConfig(seed=seed, days=days, planted_effect=planted), db_path=db)
                report = evaluate(db_path=db, predict_change=predict_change)
                scores = {m["model"]: m for m in report["models"]}
                base = scores[BASE_MODEL]
                gain = lambda m, key: 100 * (base[key] - m[key]) / base[key]  # noqa: E731
                for name in RAIL_MODELS:
                    m = scores[name]
                    rows.append({
                        "seed": seed,
                        "world": "planted" if planted else "placebo",
                        "model": name,
                        "rmse_gain_pct": gain(m, "rmse_sec"),
                        "disruption_rmse_gain_pct": gain(m, "disruption_rmse_sec"),
                        "disruption_mae_gain_pct": gain(m, "disruption_mae_sec"),
                    })
                log.info("seed %d, %s world: done", seed, "planted" if planted else "placebo")
    df = pd.DataFrame(rows)
    metrics = ["rmse_gain_pct", "disruption_rmse_gain_pct", "disruption_mae_gain_pct"]
    summary = df.groupby(["world", "model"])[metrics].agg(["mean", "min", "max"]).round(1)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(artifacts_dir / "sanity_check_runs.csv", index=False)
    (artifacts_dir / "sanity_check.json").write_text(json.dumps({
        "seeds": list(seeds),
        "days": days,
        "compared_against": BASE_MODEL,
        "summary": {f"{w} | {m}": {f"{metric}_{stat}": float(v) for (metric, stat), v in r.items()}
                    for (w, m), r in summary.iterrows()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(df, len(seeds), artifacts_dir / "sanity_check.png")
    return summary


def _plot(df: pd.DataFrame, n_seeds: int, path: Path) -> None:
    means = df.groupby(["world", "model"])["disruption_rmse_gain_pct"].mean()
    lo, hi = float(means.min()), float(means.max())
    fig, (left, right) = charts.figure(n_panels=2, width=9.0, height=2.6)
    fig.subplots_adjust(top=0.7, wspace=0.08)
    labels = list(RAIL_MODELS)
    charts.hbars(left, labels, [means[("planted", m)] for m in RAIL_MODELS], [charts.SERIES[0]] * 3,
                 fmt="{:+.1f}%", panel_title="Planted world: rail effect exists", value_range=(lo, hi))
    charts.hbars(right, [""] * 3, [means[("placebo", m)] for m in RAIL_MODELS], [charts.MUTED] * 3,
                 fmt="{:+.1f}%", panel_title="Placebo world: no rail effect (should be ~0)", value_range=(lo, hi))
    right.set_yticks([])
    charts.titles(fig, "Cut in forecast error (RMSE, %) during rail disruptions vs. model C (no rail information)",
                  f"Synthetic data, mean of {n_seeds} seeds. Same weather, incidents and bulletins in both worlds.")
    charts.save(fig, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("src.train").setLevel(logging.WARNING)
    logging.getLogger("src.ingest").setLevel(logging.WARNING)
    print(run().to_string())
