from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class ParsedRailBulletin:
    is_disrupted: int
    parsed_cause: str
    cause_en: str
    has_furikae_yuso: int
    is_suspended: int
    severity_score: float
    matched_keywords: list[str]


INCIDENT_RULES: list[tuple[str, str, float, list[str]]] = [
    (
        "track_injury_accident",
        "Track / Passenger Injury Incident (人身事故)",
        0.95,
        ["人身事故"],
    ),
    (
        "weather_severe",
        "Severe Weather / High Wind / Heavy Rain (強風・大雨・雷雨)",
        0.82,
        ["強風", "大雨", "雷雨", "台風", "雪", "倒木", "飛来物"],
    ),
    (
        "signal_power_fault",
        "Signal / Power Equipment Fault (信号・停電トラブル)",
        0.78,
        ["信号確認", "信号装置故障", "停電", "架線", "踏切安全確認", "踏切事故"],
    ),
    (
        "rolling_stock_fault",
        "Rolling Stock Inspection / Mechanical Fault (車両点検・故障)",
        0.70,
        ["車両点検", "車両故障", "ドア点検", "ブレーキ点検"],
    ),
    (
        "passenger_track_intrusion",
        "Sick Passenger / Track Intrusion / Safety Check (急病人救護・線路内立入)",
        0.55,
        ["急病人", "線路内立入", "線路内人立入", "荷物挟まり", "安全確認", "お客様救護"],
    ),
    (
        "cascading_delay",
        "Upstream Congestion / Cascading Delay (ダイヤ乱れ・遅延)",
        0.42,
        ["ダイヤ乱れ", "直通運転中止", "遅延", "遅れ"],
    ),
]

NORMAL_PATTERNS = re.compile(r"(平常運転|平常どおり|平常通り)")
SUSPENSION_PATTERNS = re.compile(r"(運転見合わせ|運転を見合わせ|運休)")
FURIKAE_PATTERNS = re.compile(r"(振替輸送)")


def parse_japanese_bulletin(text_ja: str, status_ja: str = "") -> ParsedRailBulletin:
    """Extract structured operational severity and cross-modal spillover signals from Japanese rail text."""
    combined = f"{status_ja} {text_ja}".strip()
    if not combined:
        return ParsedRailBulletin(0, "normal", "Normal Service", 0, 0, 0.0, [])

    matched_keywords: list[str] = []
    detected_cause = "normal"
    cause_en = "Normal Service (平常運転)"
    base_severity = 0.0

    for cause_key, label_en, sev, keywords in INCIDENT_RULES:
        hits = [kw for kw in keywords if kw in combined]
        if hits:
            matched_keywords.extend(hits)
            if detected_cause == "normal":
                detected_cause = cause_key
                cause_en = label_en
                base_severity = sev

    is_suspended = 1 if SUSPENSION_PATTERNS.search(combined) else 0
    has_furikae = 1 if FURIKAE_PATTERNS.search(combined) else 0

    if is_suspended:
        matched_keywords.append("運転見合わせ")
    if has_furikae:
        matched_keywords.append("振替輸送")

    if detected_cause == "normal" and not is_suspended and NORMAL_PATTERNS.search(combined):
        return ParsedRailBulletin(0, "normal", "Normal Service (平常運転)", 0, 0, 0.0, ["平常運転"])

    if detected_cause == "normal" and (is_suspended or has_furikae):
        detected_cause = "unspecified_disruption"
        cause_en = "Service Suspension / Transfer Activated"
        base_severity = 0.65

    is_disrupted = 1 if detected_cause != "normal" else 0
    if not is_disrupted:
        return ParsedRailBulletin(0, "normal", "Normal Service (平常運転)", 0, 0, 0.0, [])

    severity = base_severity
    if is_suspended:
        severity = min(1.0, severity + 0.12)
    if has_furikae:
        severity = min(1.0, severity + 0.15)

    return ParsedRailBulletin(
        is_disrupted=1,
        parsed_cause=detected_cause,
        cause_en=cause_en,
        has_furikae_yuso=has_furikae,
        is_suspended=is_suspended,
        severity_score=round(float(severity), 4),
        matched_keywords=list(dict.fromkeys(matched_keywords)),
    )


class BilingualBulletinEmbedder:
    """Dense semantic encoder for Japanese rail disruption bulletins.

    Uses `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) when available,
    falling back cleanly to character/subword TF-IDF + TruncatedSVD for fast, offline execution.
    """

    def __init__(self, n_components: int = 8, use_neural: bool = True, random_state: int = 42):
        self.n_components = n_components
        self.use_neural = use_neural
        self.random_state = random_state
        self._st_model = None
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=2048)
        self._svd: TruncatedSVD | None = None
        self._backend = "tfidf_svd"

    def _init_neural_if_available(self) -> bool:
        if not self.use_neural:
            return False
        if self._st_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self._backend = "sentence_transformers"
            return True
        except Exception:
            self._st_model = None
            self._backend = "tfidf_svd"
            return False

    def _raw_encode(self, texts: Sequence[str], fit: bool = False):
        cleaned = [t if (t and t.strip()) else "平常運転" for t in texts]
        if self._init_neural_if_available() and self._st_model is not None:
            return np.asarray(self._st_model.encode(cleaned, show_progress_bar=False), dtype=np.float32)

        if fit:
            return self._vectorizer.fit_transform(cleaned)
        return self._vectorizer.transform(cleaned)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        raw = self._raw_encode(texts, fit=True)
        n_features = raw.shape[1]
        k = min(self.n_components, max(1, n_features - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        reduced = self._svd.fit_transform(raw)
        if reduced.shape[1] < self.n_components:
            pad = np.zeros((reduced.shape[0], self.n_components - reduced.shape[1]), dtype=np.float32)
            reduced = np.hstack([reduced, pad])
        return np.asarray(reduced, dtype=np.float32)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("BilingualBulletinEmbedder must be fitted before calling transform().")
        raw = self._raw_encode(texts, fit=False)
        reduced = self._svd.transform(raw)
        if reduced.shape[1] < self.n_components:
            pad = np.zeros((reduced.shape[0], self.n_components - reduced.shape[1]), dtype=np.float32)
            reduced = np.hstack([reduced, pad])
        return np.asarray(reduced, dtype=np.float32)
