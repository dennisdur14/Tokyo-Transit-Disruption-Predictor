from __future__ import annotations

from pathlib import Path

from src.ingest import generate_simulated_history
from src.nlp_parser import BilingualBulletinEmbedder, parse_japanese_bulletin
from src.train import load_sql_feature_table, run_training_and_ablation


def test_japanese_bulletin_parser_categories():
    acc = parse_japanese_bulletin(
        "【小田急小田原線 運転見合わせ】新宿駅構内での人身事故の影響で運転を見合わせています。振替輸送を実施しています。",
        "運転見合わせ",
    )
    assert acc.is_disrupted == 1
    assert acc.parsed_cause == "track_injury_accident"
    assert acc.has_furikae_yuso == 1
    assert acc.is_suspended == 1
    assert acc.severity_score >= 0.95

    normal = parse_japanese_bulletin("【JR山手線】現在、平常どおり運転しています。", "平常運転")
    assert normal.is_disrupted == 0
    assert normal.parsed_cause == "normal"
    assert normal.severity_score == 0.0


def test_bilingual_embedder_shape():
    embedder = BilingualBulletinEmbedder(n_components=6, use_neural=False)
    texts = [
        "現在、平常どおり運転しています。",
        "人身事故の影響で運転を見合わせています。振替輸送を実施中。",
        "車両点検の影響で一部列車に遅れが出ています。",
    ]
    mat = embedder.fit_transform(texts)
    assert mat.shape == (3, 6)
    transformed = embedder.transform(["強風による速度規制でダイヤ乱れ。"])
    assert transformed.shape == (1, 6)


def test_end_to_end_sql_and_ablation(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'test_transit.db'}"

    generate_simulated_history(db_url=db_url, days=5, seed=123)
    df = load_sql_feature_table(db_url=db_url)
    assert len(df) > 1000
    assert "rolling_60m_delay_sec" in df.columns
    assert "hub_furikae_active" in df.columns

    report = run_training_and_ablation(db_url=db_url, artifacts_dir=tmp_path / "artifacts")
    assert len(report["tiers"]) == 3
    assert report["tiers"][2]["rmse_sec"] < report["tiers"][0]["rmse_sec"]
    assert report["tiers"][2]["disruption_window_rmse_sec"] < report["tiers"][1]["disruption_window_rmse_sec"]
