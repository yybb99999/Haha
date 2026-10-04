from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from attacks.common.config import (
    assert_same_mechanism,
    load_json,
    mechanism_fingerprint,
    save_json,
    sha256_file,
)
from attacks.common.logits import logit_scaled_confidence, load_logits
from attacks.common.metrics import bootstrap_attack_metrics, evaluate_attack_scores
from attacks.lira.score_offline_lira import offline_lira_scores


NON_DP_POSITIVE_CONTROL_ROLE = "attack_power_positive_control_only"
NON_DP_ALIGNED_BASELINE_ROLE = "parameter_aligned_non_dp_baseline"


def _validate_non_dp_experiment_role(config: dict) -> dict | None:
    if config["train"].get("training_mode") != "non_dp":
        return None

    role = config.get("experiment_role")
    direct_comparison = bool(config.get("direct_comparison_allowed", False))

    if role == NON_DP_POSITIVE_CONTROL_ROLE:
        if direct_comparison:
            raise RuntimeError(
                "A Non-DP positive control cannot be labeled as a direct comparison."
            )

        return None

    if role != NON_DP_ALIGNED_BASELINE_ROLE:
        raise RuntimeError(
            "Non-DP audits must be labeled as an attack-power positive control "
            "or a parameter-aligned Non-DP baseline."
        )

    if not direct_comparison:
        raise RuntimeError(
            "A parameter-aligned Non-DP baseline must explicitly allow comparison."
        )

    alignment = config.get("comparison_alignment")

    if not isinstance(alignment, dict):
        raise RuntimeError("Aligned Non-DP baseline requires comparison_alignment.")

    expected = alignment.get("expected_train_values")

    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(
            "comparison_alignment.expected_train_values must be a non-empty object."
        )

    train_config = config["train"]
    mismatches = {
        key: {"expected": value, "actual": train_config.get(key)}
        for key, value in expected.items()
        if train_config.get(key) != value
    }

    if mismatches:
        raise RuntimeError(f"Aligned Non-DP configuration mismatch: {mismatches}")

    reference_paths = alignment.get("reference_configs")

    if not isinstance(reference_paths, list) or not reference_paths:
        raise RuntimeError(
            "comparison_alignment.reference_configs must list formal audit configs."
        )

    reference_fields = alignment.get(
        "reference_config_fields",
        [
            "algorithm",
            "dataset_name",
            "lr",
            "momentum",
            "batch_size",
            "audit_train_pool",
        ],
    )
    max_step_difference = int(alignment.get("max_step_difference", 0))
    validated_references = []

    for reference_path_value in reference_paths:
        reference_path = Path(reference_path_value)

        if not reference_path.is_file():
            raise RuntimeError(f"Missing aligned reference config: {reference_path}")

        reference_config = load_json(reference_path)
        reference_train = reference_config["train"]
        reference_mismatches = {
            field: {
                "non_dp": train_config.get(field),
                "reference": reference_train.get(field),
            }
            for field in reference_fields
            if train_config.get(field) != reference_train.get(field)
        }

        if reference_mismatches:
            raise RuntimeError(
                f"Non-DP/reference alignment mismatch for {reference_path}: "
                f"{reference_mismatches}"
            )

        if config["split_manifest_path"] != reference_config.get(
            "split_manifest_path"
        ):
            raise RuntimeError(
                f"Non-DP/reference split mismatch for {reference_path}."
            )

        metrics_path = Path(reference_config["output_dir"]) / "target_metrics.json"

        if not metrics_path.is_file():
            raise RuntimeError(f"Missing aligned reference metrics: {metrics_path}")

        reference_metrics = load_json(metrics_path)
        reference_steps = int(reference_metrics.get("accounted_steps", -1))

        if abs(int(train_config["target_steps"]) - reference_steps) > max_step_difference:
            raise RuntimeError(
                f"Non-DP/reference step mismatch for {reference_path}: "
                f"non_dp={train_config['target_steps']}, reference={reference_steps}."
            )

        validated_references.append(
            {
                "config_path": str(reference_path),
                "config_sha256": sha256_file(reference_path),
                "target_metrics_path": str(metrics_path),
                "target_metrics_sha256": sha256_file(metrics_path),
                "accounted_steps": reference_steps,
            }
        )

    validated_alignment = dict(alignment)
    validated_alignment["validated_reference_configs"] = validated_references
    return validated_alignment


def _npz_scalar_text(payload: dict, key: str) -> str:
    if key not in payload:
        raise RuntimeError(f"Logits payload is missing metadata field {key!r}.")

    value = np.asarray(payload[key])

    if value.size != 1:
        raise RuntimeError(f"Metadata field {key!r} must be scalar.")

    return str(value.reshape(-1)[0])


def _load_checkpoint_args(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Audit checkpoint lacks model_state_dict: {path}")

    args = payload.get("args")

    if not isinstance(args, dict):
        raise RuntimeError(f"Audit checkpoint lacks serialized args: {path}")

    return args


def _validate_logits_payload(
    path: Path,
    payload: dict,
    *,
    candidate_indices: np.ndarray,
    expected_labels: np.ndarray | None,
    expected_candidate_pool: str,
    expected_query_augmentations: list[str] | tuple[str, ...] = ("identity",),
) -> np.ndarray:
    required = {
        "logits",
        "labels",
        "query_indices",
        "checkpoint_path",
        "checkpoint_sha256",
        "indices_path",
        "indices_sha256",
        "candidate_pool",
    }
    missing = required - set(payload)

    if missing:
        raise RuntimeError(f"{path} is missing metadata fields: {sorted(missing)}")

    query_indices = np.asarray(payload["query_indices"], dtype=np.int64).reshape(-1)

    if not np.array_equal(query_indices, candidate_indices):
        raise RuntimeError(f"Query order mismatch: {path}")

    logits = np.asarray(payload["logits"])
    labels = np.asarray(payload["labels"], dtype=np.int64).reshape(-1)

    if logits.ndim not in (2, 3) or logits.shape[0] != len(candidate_indices):
        raise RuntimeError(f"Unexpected logits shape {logits.shape}: {path}")

    if logits.shape[-1] < 2:
        raise RuntimeError(f"Logits require at least two classes: {path}")

    if "query_augmentations" in payload:
        actual_query_augmentations = [
            str(item)
            for item in np.asarray(payload["query_augmentations"]).reshape(-1)
        ]
    else:
        actual_query_augmentations = ["identity"]

    expected_query_augmentations = [
        str(item) for item in expected_query_augmentations
    ]

    if actual_query_augmentations != expected_query_augmentations:
        raise RuntimeError(
            f"Query-augmentation mismatch in {path}: "
            f"expected={expected_query_augmentations}, "
            f"actual={actual_query_augmentations}"
        )

    if logits.ndim == 2 and len(expected_query_augmentations) != 1:
        raise RuntimeError(f"Missing augmentation axis in {path}")

    if logits.ndim == 3 and logits.shape[1] != len(expected_query_augmentations):
        raise RuntimeError(f"Unexpected augmentation count in {path}")

    if labels.shape != (len(candidate_indices),):
        raise RuntimeError(f"Unexpected labels shape {labels.shape}: {path}")

    if expected_labels is not None and not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Labels mismatch: {path}")

    if not np.all(np.isfinite(logits)):
        raise RuntimeError(f"Logits contain NaN or Infinity: {path}")

    checkpoint_path = Path(_npz_scalar_text(payload, "checkpoint_path"))
    indices_path = Path(_npz_scalar_text(payload, "indices_path"))

    if not checkpoint_path.is_file() or not indices_path.is_file():
        raise RuntimeError(f"Embedded audit artifact path no longer exists: {path}")

    if sha256_file(checkpoint_path) != _npz_scalar_text(payload, "checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")

    if sha256_file(indices_path) != _npz_scalar_text(payload, "indices_sha256"):
        raise RuntimeError(f"Query-index hash mismatch: {path}")

    if _npz_scalar_text(payload, "candidate_pool") != expected_candidate_pool:
        raise RuntimeError(f"Candidate-pool mismatch: {path}")

    return labels


def _validate_runtime_metrics(
    config: dict,
    split_manifest: dict,
    output_dir: Path,
    expected_k: int,
) -> dict:
    expected_train_size = int(split_manifest["target_train_size"])
    target_metrics = load_json(output_dir / "target_metrics.json")
    refs_dir = output_dir / "references"
    reference_metrics = [
        load_json(refs_dir / f"reference_{ref_id:03d}_metrics.json")
        for ref_id in range(expected_k)
    ]
    all_metrics = [target_metrics, *reference_metrics]
    expected_q = float(config["train"]["batch_size"]) / expected_train_size
    is_non_dp = config["train"].get("training_mode") == "non_dp"
    is_fixed_steps = config["train"].get("stop_rule") == "fixed_steps"

    for index, metrics in enumerate(all_metrics):
        label = "target" if index == 0 else f"reference_{index - 1:03d}"

        if int(metrics.get("train_size", -1)) != expected_train_size:
            raise RuntimeError(f"Wrong train_size for {label}.")

        if int(metrics.get("batch_size", -1)) != int(config["train"]["batch_size"]):
            raise RuntimeError(f"Wrong batch_size for {label}.")

        if not np.isclose(float(metrics.get("sample_rate", -1.0)), expected_q):
            raise RuntimeError(f"Wrong sample_rate for {label}.")

        if is_non_dp:
            if metrics.get("epsilon") is not None:
                raise RuntimeError(f"Non-DP epsilon must be null for {label}.")

            if metrics.get("privacy_guarantee") != "none":
                raise RuntimeError(f"Wrong Non-DP privacy label for {label}.")

            if int(metrics.get("accounted_steps", -1)) != 0:
                raise RuntimeError(f"Non-DP accounted_steps must be zero for {label}.")

            if int(metrics.get("actual_dp_updates", -1)) != 0:
                raise RuntimeError(f"Non-DP actual_dp_updates must be zero for {label}.")

            if int(metrics.get("training_steps", -1)) != int(
                metrics.get("actual_updates", -2)
            ):
                raise RuntimeError(f"Non-DP training/update-step mismatch for {label}.")
        elif is_fixed_steps:
            if int(metrics.get("accounted_steps", -1)) != int(
                metrics.get("actual_dp_updates", -2)
            ):
                raise RuntimeError(f"Accounting/update-step mismatch for {label}.")

            if int(metrics.get("accounted_steps", -1)) != int(
                config["train"]["target_steps"]
            ):
                raise RuntimeError(f"Wrong fixed-step count for {label}.")
        else:
            if float(metrics.get("epsilon", np.inf)) > float(config["train"]["epsilon"]) + 1e-10:
                raise RuntimeError(f"Privacy budget exceeded for {label}.")

            if int(metrics.get("accounted_steps", -1)) != int(
                metrics.get("actual_dp_updates", -2)
            ):
                raise RuntimeError(f"Accounting/update-step mismatch for {label}.")

    if is_non_dp:
        target_steps = int(config["train"]["target_steps"])

        if any(int(item.get("training_steps", -1)) != target_steps for item in all_metrics):
            raise RuntimeError("Non-DP target/reference training_steps mismatch.")
    elif is_fixed_steps:
        target_steps = int(config["train"]["target_steps"])

        if any(int(item.get("accounted_steps", -1)) != target_steps for item in all_metrics):
            raise RuntimeError("Fixed-step target/reference accounted_steps mismatch.")
    elif config["train"]["accountant"] == "projected_gmm_pld":
        target_steps = int(config["train"]["target_steps"])

        if any(int(item.get("accounted_steps", -1)) != target_steps for item in all_metrics):
            raise RuntimeError("StepMix target/reference accounted_steps mismatch.")
    else:
        steps = {int(item.get("accounted_steps", -1)) for item in all_metrics}

        if len(steps) != 1:
            raise RuntimeError("Gaussian target/reference accounted_steps mismatch.")

    return {
        "training_mode": config["train"].get("training_mode", "dp"),
        "expected_sample_rate": expected_q,
        "target_accounted_steps": int(target_metrics["accounted_steps"]),
        "reference_accounted_steps": [
            int(item["accounted_steps"]) for item in reference_metrics
        ],
        "target_training_steps": int(target_metrics.get("training_steps", 0)),
        "reference_training_steps": [
            int(item.get("training_steps", 0)) for item in reference_metrics
        ],
    }


def _validate_privacy_certificate(config: dict, train_size: int) -> dict | None:
    if config["train"].get("training_mode") == "non_dp":
        if config.get("privacy_certificate_path") or config.get(
            "require_privacy_certificate",
            False,
        ):
            raise RuntimeError("Non-DP sanity controls must not declare a privacy certificate.")

        return None

    certificate_path = config.get("privacy_certificate_path")
    required = bool(config.get("require_privacy_certificate", False))

    if not certificate_path:
        if required:
            raise RuntimeError("A privacy certificate is required but not configured.")

        return None

    path = Path(certificate_path)
    certificate = load_json(path)
    accountant = config["train"].get("accountant")
    expected = {
        "dataset_name": config["train"]["dataset_name"],
        "epsilon": config["train"]["epsilon"],
        "target_delta": config["train"]["delta"],
        "train_size": int(train_size),
        "batch_size": config["train"]["batch_size"],
        "sample_rate": config["train"]["batch_size"] / float(train_size),
        "target_steps": config["train"]["target_steps"],
        "C_t": config["train"]["C_t"],
        "accountant": accountant,
    }

    if accountant == "projected_gmm_pld":
        projected_delta = certificate.get("projected_delta")

        if projected_delta is None or not np.isfinite(float(projected_delta)):
            raise RuntimeError("Privacy certificate has no finite projected_delta.")

        if float(projected_delta) > float(certificate["target_delta"]):
            raise RuntimeError("Privacy certificate does not satisfy target_delta.")

        expected.update(
            {
                "sigma_small": config["train"]["sigma_t"],
                "sigma_large": config["train"]["sigma_large"],
                "p_large": config["train"]["p_large"],
                "mixpld_mode": config["train"]["mixpld_mode"],
            }
        )
        validation_result = {"projected_delta": float(projected_delta)}
    elif accountant in {"rdp", "coin_aware_rdp"}:
        achieved_epsilon = certificate.get("achieved_epsilon")

        if achieved_epsilon is None or not np.isfinite(float(achieved_epsilon)):
            raise RuntimeError("RDP certificate has no finite achieved_epsilon.")

        if float(achieved_epsilon) > float(config["train"]["epsilon"]) + 1e-10:
            raise RuntimeError("RDP certificate exceeds the epsilon budget.")

        if accountant == "rdp":
            expected["sigma"] = config["train"]["sigma_t"]
        else:
            expected.update(
                {
                    "sigma_small": config["train"]["sigma_t"],
                    "sigma_large": config["train"]["sigma_large"],
                    "p_large": config["train"]["p_large"],
                    "mixpld_mode": config["train"]["mixpld_mode"],
                }
            )
        validation_result = {"achieved_epsilon": float(achieved_epsilon)}
    else:
        raise RuntimeError(f"Unsupported privacy-certificate accountant: {accountant}")

    for key, value in expected.items():
        actual = certificate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = actual is not None and np.isclose(float(actual), float(value))
        else:
            matches = actual == value

        if not matches:
            raise RuntimeError(f"Privacy certificate mismatch on {key}.")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **validation_result,
    }


def _validate_audit(config: dict, split_manifest: dict, output_dir: Path):
    lira_config = config.get("lira", {})
    is_non_dp = config["train"].get("training_mode") == "non_dp"
    comparison_alignment = _validate_non_dp_experiment_role(config)

    expected_k = int(lira_config.get("expected_k_refs", split_manifest["num_references"]))

    if expected_k != int(split_manifest["num_references"]):
        raise RuntimeError("Configured K does not match split manifest.")

    export_config = config.get("export", {})
    logits_subdir = export_config.get("logits_subdir")
    logits_dir = output_dir / logits_subdir if logits_subdir else output_dir
    refs_dir = output_dir / "references"
    logits_refs_dir = logits_dir / "references"
    expected_names = {
        f"reference_{index:03d}_logits.npz" for index in range(expected_k)
    }
    actual_paths = sorted(logits_refs_dir.glob("reference_*_logits.npz"))
    actual_names = {path.name for path in actual_paths}

    if actual_names != expected_names:
        raise RuntimeError(
            "Reference logits set is incomplete: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    candidate_indices = np.load(split_manifest["candidate_indices_path"]).astype(np.int64)
    target_train_indices = np.load(
        split_manifest["target_train_indices_path"]
    ).astype(np.int64)
    membership_labels = np.load(
        split_manifest["membership_labels_path"]
    ).astype(np.int64).reshape(-1)
    reference_keep = np.load(
        split_manifest["reference_keep_path"]
    ).astype(bool)

    if np.unique(candidate_indices).size != candidate_indices.size:
        raise RuntimeError("candidate_indices contain duplicates.")

    expected_membership = np.isin(candidate_indices, target_train_indices).astype(np.int64)

    if not np.array_equal(expected_membership, membership_labels):
        raise RuntimeError("membership_labels do not match target membership.")

    if reference_keep.shape != (expected_k, len(candidate_indices)):
        raise RuntimeError("reference_keep has an invalid shape.")

    require_all_out = bool(lira_config.get("require_all_candidates_out", True))

    if require_all_out and np.any(reference_keep):
        raise RuntimeError("OUT-only audit found an IN reference candidate.")

    reference_paths = split_manifest["reference_train_indices_paths"]

    if len(reference_paths) != expected_k:
        raise RuntimeError("Reference training-index path count mismatch.")

    for ref_id, indices_path in enumerate(reference_paths):
        reference_indices = np.load(indices_path).astype(np.int64)

        if len(reference_indices) != int(split_manifest["reference_train_size"]):
            raise RuntimeError(f"Reference {ref_id:03d} has the wrong train size.")

        actual_keep = np.isin(candidate_indices, reference_indices)

        if not np.array_equal(actual_keep, reference_keep[ref_id]):
            raise RuntimeError(f"reference_keep mismatch for reference {ref_id:03d}.")

        if require_all_out and np.any(actual_keep):
            raise RuntimeError(f"Reference {ref_id:03d} contains a candidate.")

    target_logits_path = logits_dir / "target_logits.npz"
    target_payload = load_logits(target_logits_path)
    expected_candidate_pool = export_config.get(
        "candidate_pool",
        "train",
    )
    expected_query_augmentations = export_config.get(
        "query_augmentations",
        ["identity"],
    )
    target_labels = _validate_logits_payload(
        target_logits_path,
        target_payload,
        candidate_indices=candidate_indices,
        expected_labels=None,
        expected_candidate_pool=expected_candidate_pool,
        expected_query_augmentations=expected_query_augmentations,
    )
    reference_payloads = []

    for path in actual_paths:
        payload = load_logits(path)
        _validate_logits_payload(
            path,
            payload,
            candidate_indices=candidate_indices,
            expected_labels=target_labels,
            expected_candidate_pool=expected_candidate_pool,
            expected_query_augmentations=expected_query_augmentations,
        )

        if payload["logits"].shape != target_payload["logits"].shape:
            raise RuntimeError(f"Target/reference logits shape mismatch: {path}")

        reference_payloads.append(payload)

    expected_train = config["train"]
    checkpoint_paths = [
        output_dir / "target_model.pth",
        *[
            refs_dir / f"reference_{ref_id:03d}.pth"
            for ref_id in range(expected_k)
        ],
    ]
    checkpoint_fingerprints = []

    for path in checkpoint_paths:
        checkpoint_args = _load_checkpoint_args(path)
        assert_same_mechanism({"train": expected_train}, checkpoint_args)
        checkpoint_fingerprints.append(mechanism_fingerprint(checkpoint_args))

    if len({item["sha256"] for item in checkpoint_fingerprints}) != 1:
        raise RuntimeError("Target/reference mechanism fingerprints differ.")

    runtime = _validate_runtime_metrics(
        config,
        split_manifest,
        output_dir,
        expected_k,
    )
    privacy_certificate = _validate_privacy_certificate(
        config,
        int(split_manifest["target_train_size"]),
    )
    validation = {
        "status": "passed",
        "experiment_role": config.get("experiment_role", "formal_method_comparison"),
        "direct_comparison_allowed": bool(
            config.get("direct_comparison_allowed", not is_non_dp)
        ),
        "comparison_alignment": comparison_alignment,
        "expected_k_refs": expected_k,
        "actual_k_refs": len(actual_paths),
        "num_candidates": int(len(candidate_indices)),
        "num_members": int(np.sum(membership_labels == 1)),
        "num_nonmembers": int(np.sum(membership_labels == 0)),
        "target_train_size": int(split_manifest["target_train_size"]),
        "reference_train_size": int(split_manifest["reference_train_size"]),
        "query_indices_match": True,
        "labels_match": True,
        "membership_labels_verified": True,
        "all_candidates_out_for_references": bool(not np.any(reference_keep)),
        "reference_keep_recomputed": True,
        "mechanism_fingerprints_match": True,
        "mechanism_fingerprint": checkpoint_fingerprints[0],
        "all_logits_finite": True,
        "query_augmentations": list(expected_query_augmentations),
        "score_modes": lira_config.get(
            "score_modes",
            ["official_logpdf_fixed", "paper_tail_fixed"],
        ),
        "runtime_validation": runtime,
        "privacy_certificate": privacy_certificate,
        "artifact_hashes": {
            "split_manifest": sha256_file(config["split_manifest_path"]),
            "candidate_indices": sha256_file(
                split_manifest["candidate_indices_path"]
            ),
            "target_checkpoint": _npz_scalar_text(
                target_payload,
                "checkpoint_sha256",
            ),
            "target_logits": sha256_file(target_logits_path),
            "reference_checkpoints": [
                _npz_scalar_text(payload, "checkpoint_sha256")
                for payload in reference_payloads
            ],
            "reference_logits": [sha256_file(path) for path in actual_paths],
        },
        "runtime_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    return {
        "validation": validation,
        "candidate_indices": candidate_indices,
        "membership_labels": membership_labels,
        "reference_keep": reference_keep,
        "target_payload": target_payload,
        "reference_payloads": reference_payloads,
    }


def _save_roc_csv(path: Path, evaluation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roc = evaluation["roc"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])

        for fpr, tpr, threshold in zip(
            roc["fpr"],
            roc["tpr"],
            roc["thresholds"],
        ):
            writer.writerow([fpr, tpr, threshold])


def _save_roc_plots(output_dir: Path, evaluation: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[OfflineLiRA] Skip ROC plots: {exc}")
        return

    summary = evaluation["summary"]
    fpr = np.asarray(evaluation["roc"]["fpr"], dtype=np.float64)
    tpr = np.asarray(evaluation["roc"]["tpr"], dtype=np.float64)
    label = f"AUC={summary['auc']:.4f}"

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(fpr, tpr, label=label)
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", label="Random")
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_linear.png", dpi=200)
    fig.savefig(output_dir / "roc_linear.pdf")
    plt.close(fig)

    min_fpr = summary["fpr_resolution"]
    min_tpr = summary["tpr_resolution"]
    plot_fpr = np.maximum(fpr, min_fpr)
    plot_tpr = np.maximum(tpr, min_tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(plot_fpr, plot_tpr, label=label)
    ax.plot([min_fpr, 1.0], [min_fpr, 1.0], "--", label="Random")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(min_fpr, 1.0), ylim=(min_tpr, 1.0), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_loglog.png", dpi=200)
    fig.savefig(output_dir / "roc_loglog.pdf")
    plt.close(fig)


def evaluate_offline_lira(config_path: str) -> dict:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    lira_config = config.get("lira", {})
    split_manifest = load_json(config["split_manifest_path"])
    audit = _validate_audit(config, split_manifest, output_dir)
    results_dir = output_dir / lira_config.get("results_subdir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "audit_validation.json", audit["validation"])
    target_payload = audit["target_payload"]
    target_scores = logit_scaled_confidence(
        target_payload["logits"],
        target_payload["labels"],
    )
    reference_scores = np.stack(
        [
            logit_scaled_confidence(payload["logits"], payload["labels"])
            for payload in audit["reference_payloads"]
        ],
        axis=0,
    )
    score_modes = lira_config.get(
        "score_modes",
        ["official_logpdf_fixed", "paper_tail_fixed"],
    )
    bootstrap_repeats = int(lira_config.get("bootstrap_repeats", 2000))
    bootstrap_seed = int(lira_config.get("bootstrap_seed", 20260710))
    summaries = {}

    for mode_index, score_mode in enumerate(score_modes):
        mode_dir = results_dir / score_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        lira = offline_lira_scores(
            target_scores=target_scores,
            reference_scores=reference_scores,
            reference_keep=audit["reference_keep"],
            score_mode=score_mode,
            min_std=float(lira_config.get("min_std", 1e-6)),
            min_out_references=int(
                lira_config.get("min_out_references", reference_scores.shape[0])
            ),
            require_all_out=bool(
                lira_config.get("require_all_candidates_out", True)
            ),
        )
        evaluation = evaluate_attack_scores(
            audit["membership_labels"],
            lira["attack_scores"],
        )
        bootstrap = bootstrap_attack_metrics(
            audit["membership_labels"],
            lira["attack_scores"],
            repeats=bootstrap_repeats,
            seed=bootstrap_seed + mode_index,
        )
        metrics = {
            **evaluation["summary"],
            "score_mode": score_mode,
            "center_mode": lira["center_mode"],
            "variance_mode": lira["variance_mode"],
            "out_std": float(lira["out_std"]),
            "num_references": int(reference_scores.shape[0]),
            "bootstrap": bootstrap,
        }
        save_json(mode_dir / "metrics.json", metrics)
        np.savez_compressed(
            mode_dir / "scores.npz",
            attack_scores=lira["attack_scores"],
            z_scores=lira["z_scores"],
            out_center=lira["out_center"],
            out_count=lira["out_count"],
            membership_labels=audit["membership_labels"],
            query_indices=audit["candidate_indices"],
        )
        _save_roc_csv(mode_dir / "roc.csv", evaluation)
        _save_roc_plots(mode_dir, evaluation)
        summaries[score_mode] = metrics
        print(
            f"[OfflineLiRA] mode={score_mode} AUC={metrics['auc']:.6f}, "
            f"TPR@1%FPR={metrics['tpr_at_0.01_fpr']:.6f}"
        )

    comparison_path = results_dir / "offline_lira_comparison.csv"

    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score_mode", "auc", "tpr_at_0.01_fpr", "attack_advantage"])

        for score_mode, metrics in summaries.items():
            writer.writerow(
                [
                    score_mode,
                    metrics["auc"],
                    metrics["tpr_at_0.01_fpr"],
                    metrics["attack_advantage"],
                ]
            )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audited Offline LiRA.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate_offline_lira(args.config)


if __name__ == "__main__":
    main()
