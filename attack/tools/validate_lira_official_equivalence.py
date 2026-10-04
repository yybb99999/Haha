from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _official_single_query_statistic(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    logits = np.asarray(payload["logits"])
    labels = np.asarray(payload["labels"], dtype=np.int64).reshape(-1)

    if logits.ndim != 2 or logits.shape[0] != labels.size:
        raise ValueError(f"Unexpected logits/labels shape in {path}")

    predictions = logits - np.max(logits, axis=1, keepdims=True)
    predictions = np.asarray(np.exp(predictions), dtype=np.float64)
    predictions /= np.sum(predictions, axis=1, keepdims=True)

    rows = np.arange(labels.size)
    y_true = predictions[rows, labels].copy()
    predictions[rows, labels] = 0.0
    y_wrong = np.sum(predictions, axis=1)
    return np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)


def validate(config_path: Path, *, atol: float, rtol: float) -> dict:
    config = _load_json(config_path)
    output_dir = Path(config["output_dir"])
    split_manifest = _load_json(Path(config["split_manifest_path"]))
    expected_k = int(config["lira"]["expected_k_refs"])

    target_scores = _official_single_query_statistic(
        output_dir / "target_logits.npz"
    )
    reference_scores = np.stack(
        [
            _official_single_query_statistic(
                output_dir
                / "references"
                / f"reference_{ref_id:03d}_logits.npz"
            )
            for ref_id in range(expected_k)
        ],
        axis=0,
    )

    out_center = np.median(reference_scores, axis=0)
    out_std = float(np.std(reference_scores))
    official_attack_scores = -norm.logpdf(
        target_scores,
        loc=out_center,
        scale=out_std + 1e-30,
    )

    saved_path = (
        output_dir
        / "results"
        / "official_logpdf_fixed"
        / "scores.npz"
    )
    saved = np.load(saved_path, allow_pickle=False)
    saved_attack_scores = np.asarray(saved["attack_scores"], dtype=np.float64)
    saved_out_center = np.asarray(saved["out_center"], dtype=np.float64)
    saved_metrics = _load_json(
        output_dir
        / "results"
        / "official_logpdf_fixed"
        / "metrics.json"
    )
    saved_out_std = float(saved_metrics["out_std"])

    paper_path = (
        output_dir
        / "results"
        / "paper_tail_fixed"
        / "scores.npz"
    )
    paper_saved = np.load(paper_path, allow_pickle=False)
    paper_saved_scores = np.asarray(
        paper_saved["attack_scores"], dtype=np.float64
    )
    paper_direct_scores = -log_ndtr(
        -(target_scores - out_center) / out_std
    )

    membership = np.load(
        split_manifest["membership_labels_path"], allow_pickle=False
    ).astype(np.int64)
    max_score_diff = float(
        np.max(np.abs(official_attack_scores - saved_attack_scores))
    )
    mean_score_diff = float(
        np.mean(np.abs(official_attack_scores - saved_attack_scores))
    )

    checks = {
        "scores_close": bool(
            np.allclose(
                official_attack_scores,
                saved_attack_scores,
                atol=atol,
                rtol=rtol,
            )
        ),
        "centers_close": bool(
            np.allclose(out_center, saved_out_center, atol=atol, rtol=rtol)
        ),
        "std_close": bool(
            np.isclose(out_std, saved_out_std, atol=atol, rtol=rtol)
        ),
        "paper_tail_scores_close": bool(
            np.allclose(
                paper_direct_scores,
                paper_saved_scores,
                atol=atol,
                rtol=rtol,
            )
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "config": str(config_path),
        "official_source": (
            "tensorflow/privacy/research/mi_lira_2021/plot.py:"
            "generate_ours_offline(fix_variance=True)"
        ),
        "query_mode": "single_query",
        "num_references": expected_k,
        "num_candidates": int(target_scores.size),
        "out_std_official_port": out_std,
        "out_std_saved": saved_out_std,
        "max_abs_score_difference": max_score_diff,
        "mean_abs_score_difference": mean_score_diff,
        "auc_official_port": float(
            roc_auc_score(membership, official_attack_scores)
        ),
        "auc_saved": float(roc_auc_score(membership, saved_attack_scores)),
        "paper_tail_max_abs_score_difference": float(
            np.max(np.abs(paper_direct_scores - paper_saved_scores))
        ),
        "paper_tail_auc_direct": float(
            roc_auc_score(membership, paper_direct_scores)
        ),
        "paper_tail_auc_saved": float(
            roc_auc_score(membership, paper_saved_scores)
        ),
        "checks": checks,
        "atol": float(atol),
        "rtol": float(rtol),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Offline LiRA scores with the paper's official code."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = validate(args.config, atol=args.atol, rtol=args.rtol)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")

    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
