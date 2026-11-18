"""Bulletin parser tests. Several cases are real ODPT texts or failures found in review."""
from __future__ import annotations

import pytest

from src.nlp_parser import BulletinTextEmbedder, parse_bulletin, status_category


def test_suspension_with_furikae():
    r = parse_bulletin(
        "新宿駅構内での人身事故の影響で、上下線で運転を見合わせています。現在、都営バスで振替輸送を実施しています。",
        "運転見合わせ",
    )
    assert (r.is_disrupted, r.is_suspended, r.has_furikae_yuso) == (1, 1, 1)
    assert r.parsed_cause == "track_injury_accident"
    assert r.severity_score == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "現在、１５分以上の遅延はありません。",  # real Toei text (full-width digits), 2026-11-18
        "現在、平常どおり運転しています。",
        "平常運転",
        "遅れが出ていましたが、現在は平常どおり運転しています。",  # past delay, normal now
        "振替輸送は終了しました。現在、平常どおり運転しています。",
    ],
)
def test_normal_service(text):
    r = parse_bulletin(text, "")
    assert r.is_disrupted == 0
    assert r.severity_score == 0.0


def test_resumed_after_suspension_is_not_suspended():
    r = parse_bulletin("人身事故の影響で運転を見合わせていましたが、10時15分頃に運転を再開しました。", "")
    assert r.is_disrupted == 1
    assert r.is_suspended == 0
    assert r.is_resumed == 1
    assert r.parsed_cause == "track_injury_accident"  # cause still read from the past part
    assert r.severity_score < 0.95


def test_negated_furikae():
    r = parse_bulletin("車両点検の影響で遅れが出ています。なお、振替輸送は実施しておりません。", "遅延")
    assert r.is_disrupted == 1
    assert r.has_furikae_yuso == 0
    assert r.parsed_cause == "rolling_stock_fault"


def test_snow_in_station_name_is_not_weather():
    r = parse_bulletin("雪が谷大塚駅での急病人救護の影響で、一部列車に遅れが出ています。", "")
    assert r.parsed_cause == "passenger_or_trespass"


def test_cause_without_status_words_still_counts():
    r = parse_bulletin("強風のため、速度を落として運転しています。", "")
    assert r.is_disrupted == 1
    assert r.parsed_cause == "weather_severe"


@pytest.mark.parametrize(
    "header, expected",
    [(None, "normal"), ("", "normal"), ("平常運転", "normal"), ("運転見合わせ", "suspended"),
     ("遅延", "delay"), ("ダイヤ乱れ", "delay"), ("運転再開", "resumed"), ("直通運転中止", "other")],
)
def test_status_category(header, expected):
    assert status_category(header) == expected


def test_embedder_shapes_and_unfitted_error():
    emb = BulletinTextEmbedder(n_components=6)
    with pytest.raises(RuntimeError):
        emb.transform(["遅延"])
    texts = ["現在、平常どおり運転しています。", "人身事故の影響で運転を見合わせています。", "車両点検の影響で遅れが出ています。"] * 3
    assert emb.fit_transform(texts).shape == (9, 6)
    assert emb.transform(["強風による速度規制でダイヤ乱れ。"]).shape == (1, 6)


def test_embedder_with_no_text_returns_zeros():
    emb = BulletinTextEmbedder(n_components=4)
    assert emb.fit_transform(["", "", ""]).sum() == 0.0
