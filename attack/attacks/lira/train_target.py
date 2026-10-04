from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from attacks.common.config import (
    assert_same_mechanism,
    build_main_command,
    load_json,
    mechanism_fingerprint,
    save_json,
    sha256_file,
)


def _validate_checkpoint(path: Path, expected_train: dict) -> None:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise RuntimeError(f"Invalid audit checkpoint: {path}")

    assert_same_mechanism({"train": expected_train}, payload["args"])


def train_target(config_path: str, log_path: str | None = None) -> dict:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = load_json(config["split_manifest_path"])
    train_config = dict(config["train"])
    target_seed = config.get("target_seed")

    if target_seed is None and config.get("require_explicit_target_seed", False):
        raise ValueError("Formal Offline LiRA runs require target_seed.")

    if target_seed is not None:
        train_config["seed"] = int(target_seed)

    train_config["train_indices_path"] = split_manifest["target_train_indices_path"]
    train_config["run_tag"] = config.get("run_tag", "target")
    model_path = output_dir / "target_model.pth"
    metrics_path = output_dir / "target_metrics.json"
    manifest_path = output_dir / "target_manifest.json"
    command_path = output_dir / "target_command.txt"
    resolved_config_path = output_dir / "target_resolved_config.json"
    train_config["save_model_path"] = str(model_path)
    train_config["metrics_output_path"] = str(metrics_path)
    fingerprint = mechanism_fingerprint({"train": train_config})
    save_json(
        resolved_config_path,
        {"train": train_config, "mechanism_fingerprint": fingerprint},
    )
    force_target = bool(config.get("force_target", False))

    if model_path.exists() and metrics_path.exists() and not force_target:
        _validate_checkpoint(model_path, train_config)
        manifest = {
            "target_model_path": str(model_path),
            "target_metrics_path": str(metrics_path),
            "target_command_path": str(command_path),
            "target_resolved_config_path": str(resolved_config_path),
            "model_sha256": sha256_file(model_path),
            "metrics_sha256": sha256_file(metrics_path),
            "mechanism_fingerprint": fingerprint,
        }
        save_json(manifest_path, manifest)
        print("[OfflineLiRA] Reusing validated target checkpoint.")
        return manifest

    if (model_path.exists() or metrics_path.exists()) and not force_target:
        raise RuntimeError(
            "Incomplete target artifacts exist; use force_target only after review."
        )

    command = build_main_command(
        config["python_exe"],
        config["project_root"],
        train_config,
    )
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    print("[OfflineLiRA] Running target command:")
    print(" ".join(command))
    if log_path is None:
        subprocess.run(command, cwd=config["project_root"], check=True)
    else:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with log_file.open("w", encoding="utf-8") as f:
            f.write("[OfflineLiRA] " + " ".join(command) + "\n")
            f.flush()
            subprocess.run(
                command,
                cwd=config["project_root"],
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    if not model_path.is_file() or not metrics_path.is_file():
        raise RuntimeError("Target training completed without required artifacts.")

    _validate_checkpoint(model_path, train_config)
    manifest = {
        "target_model_path": str(model_path),
        "target_metrics_path": str(metrics_path),
        "target_command_path": str(command_path),
        "target_resolved_config_path": str(resolved_config_path),
        "model_sha256": sha256_file(model_path),
        "metrics_sha256": sha256_file(metrics_path),
        "mechanism_fingerprint": fingerprint,
    }
    save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Offline LiRA target model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    train_target(args.config, args.log_path)


if __name__ == "__main__":
    main()
