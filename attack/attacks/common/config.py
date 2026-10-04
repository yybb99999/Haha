from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import List, Union

import numpy as np


SPEC_FINGERPRINT_FIELDS = (
    "algorithm",
    "dataset_name",
    "audit_train_pool",
    "lr",
    "momentum",
    "batch_size",
    "C_t",
    "epsilon",
    "delta",
    "accountant",
    "noise_mode",
    "mixpld_mode",
    "target_steps",
    "sigma_t",
    "sigma_large",
    "p_large",
    "large_step_update",
    "large_step_lr_scale",
    "use_scattering",
    "input_norm",
    "num_groups",
    "bn_noise_multiplier",
)

OPTIONAL_FINGERPRINT_FIELDS = (
    "training_mode",
    "optimizer_name",
    "weight_decay",
    "lr_schedule",
    "augment",
    "augmentation_mode",
    "model_arch",
    "data_profile",
    "sampling_scheme",
    "dp_backend",
    "dp_normalization",
    "weight_decay_scope",
    "poisson_steps_per_epoch",
    "stop_rule",
    "epochs",
    "ema_decay",
)

ALLOWED_RUN_DIFFERENCES = {
    "seed",
    "run_tag",
    "train_indices_path",
    "eval_indices_path",
    "save_model_path",
    "metrics_output_path",
    "device",
}


def load_json(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def save_json(path: Union[str, Path], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(payload),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_mechanism_config(config: dict) -> dict:
    source = config.get("train", config)
    normalized = {
        field: source.get(field)
        for field in SPEC_FINGERPRINT_FIELDS
    }
    normalized["use_scattering"] = bool(source.get("use_scattering", False))
    normalized["input_norm"] = source.get("input_norm")
    normalized["num_groups"] = int(source.get("num_groups", 27))
    normalized["bn_noise_multiplier"] = float(
        source.get("bn_noise_multiplier", 8.0)
    )

    for field in OPTIONAL_FINGERPRINT_FIELDS:
        if field in source:
            normalized[field] = source[field]

    if "augment" in normalized and isinstance(normalized["augment"], bool):
        normalized["augment"] = bool(normalized["augment"])

    if (
        normalized.get("noise_mode") == "gaussian"
        and normalized.get("stop_rule") != "fixed_steps"
    ):
        normalized["target_steps"] = None
        normalized["sigma_large"] = None
        normalized["p_large"] = None
        normalized["mixpld_mode"] = None
        normalized["large_step_update"] = None
        normalized["large_step_lr_scale"] = None

    return normalized


def mechanism_fingerprint(config: dict) -> dict:
    values = normalize_mechanism_config(config)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        "fields": values,
        "sha256": digest,
    }


def assert_same_mechanism(base_config: dict, other_config: dict) -> None:
    base = normalize_mechanism_config(base_config)
    other = normalize_mechanism_config(other_config)
    fields = sorted(set(base) | set(other))

    mismatches = {
        field: {"expected": base.get(field), "actual": other.get(field)}
        for field in fields
        if base.get(field) != other.get(field)
    }

    if mismatches:
        raise ValueError(f"Mechanism fingerprint mismatch: {mismatches}")


def build_main_command(
    python_exe: str,
    project_root: Union[str, Path],
    train_config: dict,
) -> List[str]:
    training_mode = train_config.get("training_mode", "dp")
    accountant = train_config.get("accountant")

    if training_mode == "non_dp":
        if accountant != "none":
            raise ValueError("training_mode='non_dp' requires accountant='none'.")

        command = [
            python_exe,
            "-u",
            "-m",
            "attacks.lira.train_non_dp",
        ]
    elif training_mode == "paper_dp":
        if accountant != "rdp":
            raise ValueError("training_mode='paper_dp' requires accountant='rdp'.")

        command = [
            python_exe,
            "-u",
            "-m",
            "attacks.lira.train_paper_dpsgd",
        ]
    elif training_mode == "paper_stepmix":
        if accountant not in {"projected_gmm_pld", "coin_aware_rdp"}:
            raise ValueError(
                "training_mode='paper_stepmix' requires "
                "accountant='projected_gmm_pld' or 'coin_aware_rdp'."
            )

        command = [
            python_exe,
            "-u",
            "-m",
            "attacks.lira.train_paper_stepmix",
        ]
    else:
        if accountant == "none":
            raise ValueError("accountant='none' is valid only for non-DP audit training.")

        command = [
            python_exe,
            "-u",
            str(Path(project_root) / "main.py"),
        ]

    for key, value in train_config.items():
        if value is None:
            continue

        flag = f"--{key}"

        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue

        if isinstance(value, str) and value.lower() in {"true", "false"}:
            if value.lower() == "true":
                command.append(flag)
            continue

        command.extend([flag, str(value)])

    return command
