"""Rule-based reader for Japanese rail service bulletins (ODPT odpt:TrainInformation).

Turns text such as 「人身事故の影響で運転を見合わせています。振替輸送を行っています。」 into
structured fields. It is deliberately simple and transparent -- keyword rules, not a learned
model -- so every output can be traced to the words that caused it. Accuracy on real
bulletins is measured against hand labels with `python -m src.parser_eval`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class ParsedBulletin:
    is_disrupted: int
    parsed_cause: str
    cause_en: str
    is_suspended: int  # trains are not running (on at least part of the line) right now
    is_resumed: int  # service restarted after a suspension; delays may linger
    has_furikae_yuso: int  # 振替輸送: other operators currently accept this line's tickets
    severity_score: float  # hand-set heuristic in [0, 1], see _severity()
    matched_keywords: tuple[str, ...] = ()


# (cause key, English label, base severity, keywords), ordered from most to least severe;
# the first group with a hit wins. The severities are hand-set guesses, not fitted values.
CAUSE_RULES: list[tuple[str, str, float, list[str]]] = [
    ("track_injury_accident", "Person struck by a train (人身事故)", 0.95, ["人身事故"]),
    ("weather_severe", "Severe weather (強風・大雨・大雪…)", 0.80,
     ["強風", "大雨", "雷雨", "台風", "大雪", "降雪", "積雪", "倒木", "飛来物"]),
    ("signal_power_fault", "Signal, power or overhead-wire fault", 0.75,
     ["信号", "停電", "架線", "変電所", "踏切"]),
    ("rolling_stock_fault", "Train inspection or mechanical fault (車両点検・故障)", 0.65,
     ["車両点検", "車両故障", "車両の故障", "ドア点検", "ブレーキ", "パンタグラフ"]),
    ("passenger_or_trespass", "Sick passenger, trespasser or safety check", 0.55,
     ["急病人", "お客様救護", "線路内", "立ち入り", "立入", "安全確認", "非常停止"]),
    ("congestion_knock_on", "Crowding or knock-on delay from another line", 0.40,
     ["混雑", "他社線", "乗り入れ先", "ダイヤ乱れ", "ダイヤが乱れ"]),
]
UNKNOWN_CAUSE = ("unspecified", "Disruption, cause not stated", 0.50)

# Bulletins often describe the past, then the present:
#   「…運転を見合わせていましたが、10時15分頃に運転を再開しました。」
# "…ていましたが" ("was …, but") ends the past part, so the *status* words (suspended,
# delayed, 振替輸送) are only read from the text after the last such marker. The *cause*
# is read from the whole text, because it is usually only mentioned in the past part.
PAST_CLAUSE_END = re.compile(r"(?:ていましたが|でしたが|おりましたが)[、,]?")

NORMAL = re.compile(
    r"平常(?:どおり|通り)|平常運転|(?:遅延|遅れ)(?:は|が)?(?:ありません|ございません|出ていません)|情報はありません"
)
SUSPENDED = re.compile(r"(?:運転|運行)(?:を)?見合わせ(?!て(?:い|おり)ました)")
RESUMED = re.compile(r"(?:運転|運行)(?:を)?再開")
DELAYED = re.compile(r"遅れ|遅延|ダイヤ(?:が)?乱れ|運休|直通運転(?:を)?中止|折(?:り)?返し運転")
FURIKAE = re.compile(r"振替輸送")
FURIKAE_OFF = re.compile(
    r"振替輸送(?:は|を)?(?:行って|実施して)(?:おりません|いません)"
    r"|振替輸送(?:は|を)?(?:終了|中止)"
    r"|振替輸送(?:の)?(?:実施|ご案内)は(?:ありません|ございません)"
)


def normalize(text: str | None) -> str:
    """NFKC turns full-width digits/letters (１５分, ＪＲ) into plain ASCII."""
    return unicodedata.normalize("NFKC", text or "").strip()


def _current_part(text: str) -> str:
    return PAST_CLAUSE_END.split(text)[-1]


def _find_cause(text: str) -> tuple[str, str, float, list[str]]:
    for key, label, base, keywords in CAUSE_RULES:
        hits = [kw for kw in keywords if kw in text]
        if hits:
            return key, label, base, hits
    return (*UNKNOWN_CAUSE, [])


def _severity(base: float, suspended: bool, resumed: bool, furikae: bool) -> float:
    sev = base
    if suspended:
        sev += 0.15
    if furikae:
        sev += 0.15
    if resumed and not suspended:
        sev *= 0.6  # recovering, not stopped
    return round(min(1.0, sev), 4)


_NORMAL_RESULT = ParsedBulletin(0, "normal", "Normal service", 0, 0, 0, 0.0)


def parse_bulletin(text_ja: str | None, status_ja: str | None = "") -> ParsedBulletin:
    """Read one bulletin (free text + optional short status header)."""
    text, status = normalize(text_ja), normalize(status_ja)
    if not text and not status:
        return _NORMAL_RESULT

    now = _current_part(text)
    now_with_status = f"{status} {now}"

    says_normal = bool(NORMAL.search(now))
    suspended = bool(SUSPENDED.search(now_with_status))
    resumed = bool(RESUMED.search(now_with_status)) and not suspended
    delayed = bool(DELAYED.search(status)) or (bool(DELAYED.search(now)) and not says_normal)
    furikae = bool(FURIKAE.search(now_with_status)) and not FURIKAE_OFF.search(now_with_status)

    if says_normal and not (suspended or furikae or DELAYED.search(status)):
        return _NORMAL_RESULT

    cause, label, base, hits = _find_cause(f"{status} {text}")
    disrupted = suspended or delayed or resumed or furikae or cause != UNKNOWN_CAUSE[0]
    if not disrupted:
        return _NORMAL_RESULT

    flags = [("運転見合わせ", suspended), ("運転再開", resumed), ("遅れ・遅延", delayed), ("振替輸送", furikae)]
    return ParsedBulletin(
        is_disrupted=1,
        parsed_cause=cause,
        cause_en=label,
        is_suspended=int(suspended),
        is_resumed=int(resumed),
        has_furikae_yuso=int(furikae),
        severity_score=_severity(base, suspended, resumed, furikae),
        matched_keywords=tuple(hits) + tuple(name for name, on in flags if on),
    )


def status_category(status_ja: str | None) -> str:
    """Bucket the short status header alone -- no reading of the free text.

    This is the "structured data only" baseline the text parser has to beat.
    """
    s = normalize(status_ja)
    if not s or "平常" in s:
        return "normal"
    # 直通運転中止 = through-service to another company's line is cut, but this line runs.
    if "見合わせ" in s or re.search(r"(?<!直通)運転中止", s):
        return "suspended"
    if "再開" in s:
        return "resumed"
    if any(word in s for word in ("遅延", "遅れ", "乱れ")):
        return "delay"
    return "other"


class BulletinTextEmbedder:
    """Character n-gram TF-IDF compressed with SVD into a few dense columns.

    Japanese has no spaces between words, so character n-grams work without a word
    tokenizer. Fit on training rows only, then reuse the fitted vocabulary on test rows.
    """

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2, max_features=4000)
        self.svd: TruncatedSVD | None = None
        self.fitted = False

    @staticmethod
    def _clean(texts: Sequence[str]) -> list[str]:
        return [normalize(t) or "平常" for t in texts]

    def _pad(self, mat: np.ndarray) -> np.ndarray:
        out = np.zeros((mat.shape[0], self.n_components), dtype=np.float32)
        out[:, : mat.shape[1]] = mat[:, : self.n_components]
        return out

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = self._clean(texts)
        try:
            tfidf = self.vectorizer.fit_transform(cleaned)
        except ValueError:  # vocabulary empty (e.g. no disruptions at all in training data)
            self.fitted = True
            return np.zeros((len(cleaned), self.n_components), dtype=np.float32)
        self.fitted = True
        if tfidf.shape[1] < 2:
            return np.zeros((len(cleaned), self.n_components), dtype=np.float32)
        k = min(self.n_components, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        return self._pad(self.svd.fit_transform(tfidf))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit_transform() on training texts first.")
        cleaned = self._clean(texts)
        if self.svd is None:
            return np.zeros((len(cleaned), self.n_components), dtype=np.float32)
        return self._pad(self.svd.transform(self.vectorizer.transform(cleaned)))
