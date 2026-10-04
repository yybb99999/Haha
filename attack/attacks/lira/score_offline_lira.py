from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from scipy.special import log_ndtr

from attacks.common.logits import logit_scaled_confidence, load_logits
from attacks.common.metrics import evaluate_attack_scores, save_attack_metrics


VALID_SCORE_MODES = {
    "official_logpdf_fixed",
    "paper_tail_fixed",
    "legacy_tail_fixed_mean",
}


def _validate_lira_inputs(
    target_scores,
    reference_scores,
    reference_keep=None,
    *,
    min_out_references: int = 2,
):
    target_scores = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)

    if reference_scores.ndim != 2:
        raise ValueError(
            f"reference_scores must have shape [K, N], got {reference_scores.shape}"
        )

    num_references, num_samples = reference_scores.shape

    if target_scores.shape != (num_samples,):
        raise ValueError(
            f"target_scores shape {target_scores.shape} does not match N={num_samples}"
        )

    if min_out_references <= 0 or min_out_references > num_references:
        raise ValueError("min_out_references must be in [1, K].")

    if not np.all(np.isfinite(target_scores)):
        raise ValueError("target_scores contain NaN or Infinity.")

    if not np.all(np.isfinite(reference_scores)):
        raise ValueError("reference_scores contain NaN or Infinity.")

    if reference_keep is None:
        reference_keep = np.zeros((num_references, num_samples), dtype=bool)
    else:
        reference_keep = np.asarray(reference_keep, dtype=bool)

        if reference_keep.shape != reference_scores.shape:
            raise ValueError("reference_keep must have shape [K, N].")

    out_mask = ~reference_keep
    out_counts = np.sum(out_mask, axis=0)

    if np.any(out_counts < min_out_references):
        bad = np.flatnonzero(out_counts < min_out_references)
        raise ValueError(
            f"{len(bad)} candidates have fewer than {min_out_references} "
            "OUT reference models."
        )

    return target_scores, reference_scores, reference_keep, out_mask, out_counts


def _estimate_official_fixed_out_distribution(
    reference_scores: np.ndarray,
    out_mask: np.ndarray,
    *,
    min_std: float,
):
    centers = np.empty(reference_scores.shape[1], dtype=np.float64)
    pooled_out_values = []

    for index in range(reference_scores.shape[1]):
        values = reference_scores[out_mask[:, index], index]
        centers[index] = float(np.median(values))
        pooled_out_values.append(values)

    fixed_std = float(np.std(np.concatenate(pooled_out_values), ddof=0))

    if not np.isfinite(fixed_std):
        raise ValueError("Estimated OUT standard deviation is non-finite.")

    return centers, max(fixed_std, float(min_std))


def _estimate_legacy_fixed_out_distribution(
    reference_scores: np.ndarray,
    out_mask: np.ndarray,
    *,
    min_std: float,
):
    centers = np.empty(reference_scores.shape[1], dtype=np.float64)
    residuals = []

    for index in range(reference_scores.shape[1]):
        values = reference_scores[out_mask[:, index], index]
        centers[index] = float(np.mean(values))
        residuals.append(values - centers[index])

    pooled = np.concatenate(residuals)
    fixed_std = float(np.std(pooled, ddof=1)) if pooled.size > 1 else min_std

    if not np.isfinite(fixed_std):
        raise ValueError("Estimated legacy OUT standard deviation is non-finite.")

    return centers, max(fixed_std, float(min_std))


def offline_lira_scores(
    *,
    target_scores,
    reference_scores,
    reference_keep=None,
    score_mode: str = "official_logpdf_fixed",
    min_std: float = 1e-6,
    min_out_references: int = 2,
    require_all_out: bool = False,
) -> Dict[str, np.ndarray]:
    if score_mode not in VALID_SCORE_MODES:
        raise ValueError(
            f"Unknown score_mode={score_mode!r}; expected {sorted(VALID_SCORE_MODES)}"
        )

    if min_std <= 0.0:
        raise ValueError("min_std must be positive.")

    (
        target_scores,
        reference_scores,
        reference_keep,
        out_mask,
        out_counts,
    ) = _validate_lira_inputs(
        target_scores,
        reference_scores,
        reference_keep,
        min_out_references=min_out_references,
    )
    num_references = reference_scores.shape[0]

    if require_all_out and not np.all(out_counts == num_references):
        bad = np.flatnonzero(out_counts != num_references)
        raise ValueError(
            f"OUT-only audit requires all {num_references} references to exclude "
            f"every candidate; {len(bad)} candidates violate this condition."
        )

    if score_mode == "legacy_tail_fixed_mean":
        out_center, fixed_std = _estimate_legacy_fixed_out_distribution(
            reference_scores,
            out_mask,
            min_std=min_std,
        )
        center_mode = "mean"
    else:
        out_center, fixed_std = _estimate_official_fixed_out_distribution(
            reference_scores,
            out_mask,
            min_std=min_std,
        )
        center_mode = "median"

    z_scores = (target_scores - out_center) / fixed_std
    log_tail_probability = None

    if score_mode == "official_logpdf_fixed":
        attack_scores = (
            0.5 * np.square(z_scores)
            + np.log(fixed_std)
            + 0.5 * np.log(2.0 * np.pi)
        )
    else:
        log_tail_probability = log_ndtr(-z_scores)
        attack_scores = -log_tail_probability

    if not np.all(np.isfinite(attack_scores)):
        raise ValueError("LiRA attack scores contain NaN or Infinity.")

    result = {
        "attack_scores": attack_scores.astype(np.float64),
        "target_scores": target_scores.astype(np.float64),
        "out_center": out_center.astype(np.float64),
        "out_std": np.asarray(fixed_std, dtype=np.float64),
        "z_scores": z_scores.astype(np.float64),
        "out_count": out_counts.astype(np.int64),
        "reference_keep": reference_keep.astype(bool),
        "score_mode": score_mode,
        "center_mode": center_mode,
        "variance_mode": "global_fixed",
    }

    if log_tail_probability is not None:
        result["log_tail_probability"] = log_tail_probability.astype(np.float64)

    return result


def _self_test() -> None:
    rng = np.random.default_rng(123)
    num_samples = 400
    num_refs = 8
    labels = np.r_[np.ones(num_samples // 2), np.zeros(num_samples // 2)]
    reference_scores = rng.normal(0.0, 1.0, size=(num_refs, num_samples))
    target_scores = rng.normal(0.0, 1.0, size=num_samples)
    target_scores[labels == 1] += 1.5

    for mode in sorted(VALID_SCORE_MODES):
        result = offline_lira_scores(
            target_scores=target_scores,
            reference_scores=reference_scores,
            score_mode=mode,
            min_out_references=num_refs,
            require_all_out=True,
        )
        evaluation = evaluate_attack_scores(labels, result["attack_scores"])
        auc = evaluation["summary"]["auc"]
        print(f"[OfflineLiRA self-test] mode={mode} AUC={auc:.4f}")

        if mode != "official_logpdf_fixed" and auc <= 0.65:
            raise RuntimeError(f"Offline LiRA self-test failed for {mode}.")

    extreme = offline_lira_scores(
        target_scores=np.full(num_samples, 100.0),
        reference_scores=reference_scores,
        score_mode="paper_tail_fixed",
        min_out_references=num_refs,
        require_all_out=True,
    )["attack_scores"]

    if not np.all(np.isfinite(extreme)) or np.unique(extreme).size <= 1:
        raise RuntimeError("Stable tail self-test failed.")


def _load_reference_score_matrix(paths: List[str], expected_labels) -> np.ndarray:
    scores = []

    for path in paths:
        payload = load_logits(path)

        if not np.array_equal(payload["labels"], expected_labels):
            raise ValueError(f"Reference labels do not match target labels: {path}")

        scores.append(logit_scaled_confidence(payload["logits"], payload["labels"]))

    return np.stack(scores, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Offline LiRA from logits.")
    parser.add_argument("--target_logits", type=str)
    parser.add_argument("--reference_logits", type=str, nargs="*")
    parser.add_argument("--membership_labels", type=str)
    parser.add_argument("--reference_keep", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="official_logpdf_fixed",
        choices=sorted(VALID_SCORE_MODES),
    )
    parser.add_argument("--min_out_references", type=int, default=2)
    parser.add_argument("--require_all_out", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.target_logits or not args.reference_logits or not args.membership_labels:
        raise ValueError(
            "--target_logits, --reference_logits, and --membership_labels are required."
        )

    target_payload = load_logits(args.target_logits)
    target_labels = target_payload["labels"]
    target_scores = logit_scaled_confidence(target_payload["logits"], target_labels)
    reference_scores = _load_reference_score_matrix(
        args.reference_logits,
        target_labels,
    )
    membership_labels = np.load(args.membership_labels).reshape(-1).astype(np.int64)
    reference_keep = (
        np.load(args.reference_keep).astype(bool)
        if args.reference_keep
        else None
    )
    lira = offline_lira_scores(
        target_scores=target_scores,
        reference_scores=reference_scores,
        reference_keep=reference_keep,
        score_mode=args.score_mode,
        min_out_references=args.min_out_references,
        require_all_out=args.require_all_out,
    )
    evaluation = evaluate_attack_scores(membership_labels, lira["attack_scores"])
    payload = {
        **evaluation["summary"],
        "score_mode": args.score_mode,
        "variance_mode": "global_fixed",
    }

    if args.output_json:
        save_attack_metrics(args.output_json, payload)
        print(f"[OfflineLiRA] Saved metrics to {args.output_json}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
