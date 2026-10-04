from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attacks.common.config import load_json, sha256_file
from attacks.lira import evaluate as legacy_evaluate
from privacy_analysis.PLD.CoinAwareStepMixPLD import DP_ACCOUNTING_VERSION


PLD_BACKEND = "google_dp_accounting_public_coin"
_LEGACY_VALIDATE_CERTIFICATE = legacy_evaluate._validate_privacy_certificate


def _validate_privacy_certificate(config: dict, train_size: int) -> dict | None:
    if config["train"].get("accountant") != "coin_aware_pld":
        return _LEGACY_VALIDATE_CERTIFICATE(config, train_size)
    if not config.get("require_privacy_certificate", False):
        raise RuntimeError("Formal coin_aware_pld audit requires a certificate.")
    certificate_path = config.get("privacy_certificate_path")
    if not certificate_path:
        raise RuntimeError("coin_aware_pld certificate path is missing.")

    path = Path(certificate_path)
    certificate = load_json(path)
    train = config["train"]
    projected_delta = certificate.get("projected_delta")
    achieved_epsilon = certificate.get("achieved_epsilon")
    if projected_delta is None or not np.isfinite(float(projected_delta)):
        raise RuntimeError("PLD certificate has no finite projected_delta.")
    if float(projected_delta) > float(train["delta"]):
        raise RuntimeError("PLD certificate exceeds target_delta.")
    if achieved_epsilon is None or not np.isfinite(float(achieved_epsilon)):
        raise RuntimeError("PLD certificate has no finite achieved_epsilon.")
    if float(achieved_epsilon) > float(train["epsilon"]) + 1e-10:
        raise RuntimeError("PLD certificate exceeds the epsilon budget.")
    if certificate.get("public_coin") is not True:
        raise RuntimeError("PLD certificate is not public-coin.")
    if certificate.get("pessimistic_estimate") is not True:
        raise RuntimeError("PLD certificate is not pessimistic.")
    numerical_validation = certificate.get("numerical_validation")
    if not isinstance(numerical_validation, dict):
        raise RuntimeError("PLD certificate has no numerical validation.")
    if numerical_validation.get("status") != "passed":
        raise RuntimeError("PLD certificate numerical validation did not pass.")

    expected = {
        "dataset_name": train["dataset_name"],
        "epsilon": train["epsilon"],
        "target_delta": train["delta"],
        "train_size": int(train_size),
        "batch_size": train["batch_size"],
        "sample_rate": train["batch_size"] / float(train_size),
        "target_steps": train["target_steps"],
        "C_t": train["C_t"],
        "sigma_small": train["sigma_t"],
        "sigma_large": train["sigma_large"],
        "p_large": train["p_large"],
        "mixpld_mode": train["mixpld_mode"],
        "accountant": "coin_aware_pld",
        "pld_backend": PLD_BACKEND,
        "backend_version": DP_ACCOUNTING_VERSION,
        "value_discretization_interval": train[
            "pld_value_discretization_interval"
        ],
        "log_mass_truncation_bound": train["pld_log_mass_truncation_bound"],
        "tail_mass_truncation": train["pld_tail_mass_truncation"],
    }
    for key, expected_value in expected.items():
        actual = certificate.get(key)
        if isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            matches = actual is not None and np.isclose(
                float(actual), float(expected_value)
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise RuntimeError(f"PLD certificate mismatch on {key}.")
    numerical_expected = {
        "max_single_step_absolute_mass_error": train[
            "pld_max_single_step_mass_error"
        ],
        "max_composed_absolute_mass_error": train[
            "pld_max_composed_mass_error"
        ],
    }
    for key, expected_value in numerical_expected.items():
        actual = numerical_validation.get(key)
        if actual is None or not np.isclose(float(actual), float(expected_value)):
            raise RuntimeError(f"PLD numerical gate mismatch on {key}.")
    if (
        float(numerical_validation["single_step_max_absolute_mass_error"])
        > float(train["pld_max_single_step_mass_error"])
    ):
        raise RuntimeError("PLD single-step numerical mass gate exceeded.")
    if (
        float(numerical_validation["composed_max_absolute_mass_error"])
        > float(train["pld_max_composed_mass_error"])
    ):
        raise RuntimeError("PLD composed numerical mass gate exceeded.")
    if train.get("privacy_certificate_sha256") != sha256_file(path):
        raise RuntimeError("PLD certificate SHA-256 mismatch in training config.")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "projected_delta": float(projected_delta),
        "achieved_epsilon": float(achieved_epsilon),
        "backend": PLD_BACKEND,
        "backend_version": DP_ACCOUNTING_VERSION,
    }


def evaluate_offline_lira(config_path: str) -> dict:
    previous_validator = legacy_evaluate._validate_privacy_certificate
    legacy_evaluate._validate_privacy_certificate = _validate_privacy_certificate
    try:
        return legacy_evaluate.evaluate_offline_lira(config_path)
    finally:
        legacy_evaluate._validate_privacy_certificate = previous_validator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate public-coin PLD StepMix Offline LiRA."
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate_offline_lira(args.config)


if __name__ == "__main__":
    main()
