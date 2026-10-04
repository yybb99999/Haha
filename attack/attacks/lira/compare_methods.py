from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from attacks.common.config import save_json
from attacks.common.metrics import evaluate_attack_scores


def _load_scores(output_dir: Path, score_mode: str) -> dict:
    path = output_dir / "results" / score_mode / "scores.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def paired_bootstrap_compare(
    gaussian_output_dir: str,
    stepmix_output_dir: str,
    *,
    score_mode: str = "official_logpdf_fixed",
    repeats: int = 2000,
    seed: int = 20260710,
    output_path: str | None = None,
) -> dict:
    gaussian = _load_scores(Path(gaussian_output_dir), score_mode)
    stepmix = _load_scores(Path(stepmix_output_dir), score_mode)

    for key in ("query_indices", "membership_labels"):
        if not np.array_equal(gaussian[key], stepmix[key]):
            raise RuntimeError(f"Gaussian/StepMix {key} mismatch.")

    labels = np.asarray(gaussian["membership_labels"], dtype=np.int64)
    gaussian_scores = np.asarray(gaussian["attack_scores"], dtype=np.float64)
    stepmix_scores = np.asarray(stepmix["attack_scores"], dtype=np.float64)

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    rng = np.random.default_rng(seed)
    member_indices = np.flatnonzero(labels == 1)
    nonmember_indices = np.flatnonzero(labels == 0)
    delta_auc = np.empty(repeats, dtype=np.float64)
    delta_tpr = np.empty(repeats, dtype=np.float64)

    for index in range(repeats):
        sampled = np.concatenate(
            [
                rng.choice(member_indices, size=len(member_indices), replace=True),
                rng.choice(nonmember_indices, size=len(nonmember_indices), replace=True),
            ]
        )
        gaussian_metrics = evaluate_attack_scores(
            labels[sampled],
            gaussian_scores[sampled],
            fpr_targets=(0.01,),
        )["summary"]
        stepmix_metrics = evaluate_attack_scores(
            labels[sampled],
            stepmix_scores[sampled],
            fpr_targets=(0.01,),
        )["summary"]
        delta_auc[index] = stepmix_metrics["auc"] - gaussian_metrics["auc"]
        delta_tpr[index] = (
            stepmix_metrics["tpr_at_0.01_fpr"]
            - gaussian_metrics["tpr_at_0.01_fpr"]
        )

    def interval(values):
        return {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    gaussian_point = evaluate_attack_scores(labels, gaussian_scores)["summary"]
    stepmix_point = evaluate_attack_scores(labels, stepmix_scores)["summary"]
    result = {
        "score_mode": score_mode,
        "repeats": int(repeats),
        "seed": int(seed),
        "num_candidates": int(len(labels)),
        "gaussian": gaussian_point,
        "stepmix": stepmix_point,
        "delta_stepmix_minus_gaussian": {
            "auc": interval(delta_auc),
            "tpr_at_0.01_fpr": interval(delta_tpr),
        },
    }

    if output_path:
        output_path = Path(output_path)
        save_json(output_path, result)
        csv_path = output_path.with_suffix(".csv")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean_delta", "lower_95", "upper_95"])

            for metric, values in result["delta_stepmix_minus_gaussian"].items():
                writer.writerow(
                    [metric, values["mean"], values["lower_95"], values["upper_95"]]
                )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison for Gaussian and StepMix LiRA."
    )
    parser.add_argument("--gaussian_output_dir", required=True)
    parser.add_argument("--stepmix_output_dir", required=True)
    parser.add_argument("--score_mode", default="official_logpdf_fixed")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    result = paired_bootstrap_compare(**vars(args))
    delta = result["delta_stepmix_minus_gaussian"]
    print(
        "[OfflineLiRA comparison] "
        f"delta_auc={delta['auc']['mean']:.6f} "
        f"CI=[{delta['auc']['lower_95']:.6f}, {delta['auc']['upper_95']:.6f}]"
    )


if __name__ == "__main__":
    main()
