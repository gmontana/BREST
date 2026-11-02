"""Evaluate locally held paper-cohort predictions without transmitting any data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

COHORTS = {
    "uk_hologic": {"women": 7596, "cases": 1899, "published_auc": 0.751},
    "uk_siemens": {"women": 1600, "cases": 400, "published_auc": 0.719},
    "us_hologic": {"women": 388, "cases": 97, "published_auc": 0.670},
}


def delong_interval(labels, scores):
    """Normal DeLong interval for one AUC, including half credit for ties."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("DeLong variance requires at least two observations in each class")
    placement = (positive[:, None] > negative[None, :]).astype(float)
    placement += 0.5 * (positive[:, None] == negative[None, :])
    value = float(placement.mean())
    variance = placement.mean(axis=1).var(ddof=1) / len(positive)
    variance += placement.mean(axis=0).var(ddof=1) / len(negative)
    width = float(norm.ppf(0.975) * np.sqrt(variance))
    return value, [max(0.0, value - width), min(1.0, value + width)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--label-column", default="Episode Label")
    parser.add_argument("--score-column", default="Predicted Probability")
    parser.add_argument("--id-column", default="ClientID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, dtype={args.id_column: str})
    expected = COHORTS[args.cohort]
    if args.id_column not in frame or frame[args.id_column].isna().any() or frame[args.id_column].duplicated().any():
        raise ValueError("Paper evaluation requires one identified examination per woman")
    labels = pd.to_numeric(frame[args.label_column]).to_numpy()
    scores = pd.to_numeric(frame[args.score_column]).to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and every prediction finite")
    if len(frame) != expected["women"] or int(labels.sum()) != expected["cases"]:
        raise ValueError("Cohort counts differ from the published evaluation; check the original study membership")
    auc, interval = delong_interval(labels, scores)
    assert abs(auc - roc_auc_score(labels, scores)) < 1e-12
    report = {
        "cohort": args.cohort, "women": len(frame), "cases": int(labels.sum()),
        "controls": int((labels == 0).sum()), "auc": auc, "delong_95_ci": interval,
        "published_auc": expected["published_auc"],
        "published_rounded_auc_reproduced": round(auc, 3) == expected["published_auc"],
        "membership_note": "Counts alone do not establish identical membership; use the original study cohort.",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
