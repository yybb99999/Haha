from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from attacks.common.config import save_json


def _validate_attack_inputs(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape.")

    if labels.size == 0:
        raise ValueError("labels and scores cannot be empty.")

    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("membership labels must be binary values in {0, 1}.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("attack scores contain NaN or Infinity.")

    num_members = int(np.sum(labels == 1))
    num_nonmembers = int(np.sum(labels == 0))

    if num_members == 0 or num_nonmembers == 0:
        raise ValueError("At least one member and one nonmember are required.")

    return labels, scores, num_members, num_nonmembers


def evaluate_attack_scores(
    labels,
    scores,
    fpr_targets: Iterable[float] = (0.01, 0.001),
) -> Dict:
    labels, scores, num_members, num_nonmembers = _validate_attack_inputs(
        labels,
        scores,
    )
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
        drop_intermediate=False,
    )
    auc = float(roc_auc_score(labels, scores))
    advantage = float(np.max(tpr - fpr))
    summary = {
        "auc": auc,
        "attack_advantage": advantage,
        "num_samples": int(labels.size),
        "num_members": num_members,
        "num_nonmembers": num_nonmembers,
        "fpr_resolution": float(1.0 / num_nonmembers),
        "tpr_resolution": float(1.0 / num_members),
        "fpr_targets": {},
    }

    for target_fpr in fpr_targets:
        target_fpr = float(target_fpr)

        if not (0.0 <= target_fpr <= 1.0):
            raise ValueError("FPR targets must be in [0, 1].")

        eligible = np.flatnonzero(fpr <= target_fpr)
        selected = 0 if eligible.size == 0 else int(
            eligible[np.argmax(tpr[eligible])]
        )
        threshold = thresholds[selected]
        result = {
            "target_fpr": target_fpr,
            "tpr": float(tpr[selected]),
            "observed_fpr": float(fpr[selected]),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
        }
        key = f"tpr_at_{target_fpr:g}_fpr"
        summary[key] = result["tpr"]
        summary["fpr_targets"][str(target_fpr)] = result

    return {
        "summary": summary,
        "roc": {
            "fpr": fpr.astype(np.float64),
            "tpr": tpr.astype(np.float64),
            "thresholds": thresholds.astype(np.float64),
        },
    }


def bootstrap_attack_metrics(
    labels,
    scores,
    *,
    repeats: int = 2000,
    seed: int = 20260710,
) -> dict:
    labels, scores, _, _ = _validate_attack_inputs(labels, scores)

    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive.")

    rng = np.random.default_rng(seed)
    member_indices = np.flatnonzero(labels == 1)
    nonmember_indices = np.flatnonzero(labels == 0)
    auc_values = np.empty(repeats, dtype=np.float64)
    tpr_values = np.empty(repeats, dtype=np.float64)

    for index in range(repeats):
        sampled = np.concatenate(
            [
                rng.choice(member_indices, size=len(member_indices), replace=True),
                rng.choice(nonmember_indices, size=len(nonmember_indices), replace=True),
            ]
        )
        result = evaluate_attack_scores(
            labels[sampled],
            scores[sampled],
            fpr_targets=(0.01,),
        )["summary"]
        auc_values[index] = result["auc"]
        tpr_values[index] = result["tpr_at_0.01_fpr"]

    def interval(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    return {
        "repeats": int(repeats),
        "seed": int(seed),
        "auc": interval(auc_values),
        "tpr_at_0.01_fpr": interval(tpr_values),
    }


def save_attack_metrics(path, metrics: dict) -> None:
    save_json(path, metrics)
