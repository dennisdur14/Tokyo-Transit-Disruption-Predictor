"""Measure the bulletin parser against hand labels on REAL collected bulletins.

    python -m src.parser_eval export   # add new unique bulletins to data/labels/bulletins.csv
    python -m src.parser_eval score    # compare the parser with your labels

Labelling: open data/labels/bulletins.csv (Excel / Google Sheets work; the file is UTF-8
with a BOM so Japanese displays correctly) and fill each label_* column with 1 or 0:

    label_is_disrupted   trains on this line are disrupted right now (delayed, suspended, ...)
    label_is_suspended   trains are not running on at least part of the line right now
    label_is_resumed     service restarted after a suspension (delays may linger)
    label_has_furikae    振替輸送 (ticket acceptance on other lines) is in effect right now
    label_cause          optional: one of the cause keys in src/nlp_parser.py, or "unspecified"

Leave a row blank to skip it. Decide from the text alone -- don't look at the parser's
output first, or your labels will drift towards it. Re-running `export` only appends new
bulletins; it never touches labels you have already filled in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, RAW_DIR
from src.ingest import read_snapshots
from src.nlp_parser import parse_bulletin
from src.simulate import JST

LABELS_PATH = DATA_DIR / "labels" / "bulletins.csv"
FIELDS = {  # label column -> parser attribute
    "label_is_disrupted": "is_disrupted",
    "label_is_suspended": "is_suspended",
    "label_is_resumed": "is_resumed",
    "label_has_furikae": "has_furikae_yuso",
}
COLUMNS = ["bulletin_id", "line_id", "first_seen_jst", "status_ja", "text_ja", "text_en",
           *FIELDS, "label_cause", "notes"]


def _bulletin_id(status: str, text: str) -> str:
    return hashlib.sha1(f"{status}\n{text}".encode()).hexdigest()[:12]


def export(raw_dir: Path = RAW_DIR, path: Path = LABELS_PATH) -> int:
    existing = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=COLUMNS)
    known = set(existing["bulletin_id"])
    new_rows = []
    for snap in read_snapshots(raw_dir):
        seen = datetime.fromtimestamp(snap["collected_epoch"], JST).isoformat(timespec="minutes")
        for r in snap.get("rail", []):
            status, text = r.get("status_ja") or "", r.get("text_ja") or ""
            bid = _bulletin_id(status, text)
            if bid not in known and text:
                known.add(bid)
                new_rows.append({"bulletin_id": bid, "line_id": r.get("railway") or "", "first_seen_jst": seen,
                                 "status_ja": status, "text_ja": text, "text_en": r.get("text_en") or ""})
    out = pd.concat([existing, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True).fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    out[COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(new_rows)} new bulletins added; {len(out)} in {path}")
    return len(new_rows)


def _binary_metrics(truth: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"n": len(truth), "positives": int(truth.sum()), "accuracy": round(float((truth == pred).mean()), 3),
            "precision": round(precision, 3), "recall": round(recall, 3), "false_pos": fp, "false_neg": fn}


def score(path: Path = LABELS_PATH, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.parser_eval export` and label some rows first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    labeled = df[df["label_is_disrupted"].isin(["0", "1"])].copy()
    if labeled.empty:
        raise SystemExit(f"No labelled rows yet in {path} (fill label_is_disrupted with 0 or 1).")
    parsed = [parse_bulletin(t, s) for t, s in zip(labeled["text_ja"], labeled["status_ja"])]
    report: dict = {"labelled_bulletins": len(labeled), "fields": {}, "disagreements": []}
    for col, attr in FIELDS.items():
        rows = labeled[labeled[col].isin(["0", "1"])]
        if rows.empty:
            continue
        pred = pd.Series([getattr(parsed[labeled.index.get_loc(i)], attr) for i in rows.index], index=rows.index)
        report["fields"][col.removeprefix("label_")] = _binary_metrics(rows[col].astype(int), pred)
        for i in rows.index[rows[col].astype(int) != pred]:
            report["disagreements"].append({"field": col.removeprefix("label_"), "label": int(rows.at[i, col]),
                                            "parser": int(pred[i]), "status_ja": rows.at[i, "status_ja"],
                                            "text_ja": rows.at[i, "text_ja"]})
    with_cause = labeled[(labeled["label_cause"] != "") & (labeled["label_is_disrupted"] == "1")]
    if not with_cause.empty:
        hits = [parsed[labeled.index.get_loc(i)].parsed_cause == with_cause.at[i, "label_cause"] for i in with_cause.index]
        report["fields"]["cause"] = {"n": len(with_cause), "accuracy": round(sum(hits) / len(hits), 3)}

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "parser_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Parser vs. {len(labeled)} hand-labelled real bulletins:")
    print(pd.DataFrame(report["fields"]).T.to_string())
    if report["disagreements"]:
        print(f"\n{len(report['disagreements'])} disagreements (first 15) -- use these to improve the rules:")
        for d in report["disagreements"][:15]:
            print(f"  [{d['field']}] label={d['label']} parser={d['parser']}  {d['status_ja']} | {d['text_ja'][:90]}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "score"])
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    args = parser.parse_args()
    if args.command == "export":
        export(Path(args.raw_dir), Path(args.labels))
    else:
        score(Path(args.labels))


if __name__ == "__main__":
    main()
