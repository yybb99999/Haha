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


def _checkpoint_args(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise RuntimeError(f"Invalid audit checkpoint: {path}")

    return payload["args"]


def train_references(config_path: str) -> dict:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    refs_dir = output_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = output_dir / "target_model.pth"

    if not target_model_path.is_file():
        raise FileNotFoundError(target_model_path)

    target_args = _checkpoint_args(target_model_path)
    assert_same_mechanism({"train": config["train"]}, target_args)
    target_fingerprint = mechanism_fingerprint(target_args)
    split_manifest = load_json(config["split_manifest_path"])
    reference_paths = split_manifest["reference_train_indices_paths"]
    base_seed = int(config.get("reference_seed_base", 10000))
    manifests = []

    for ref_id, indices_path in enumerate(reference_paths):
        train_config = dict(config["train"])
        train_config["train_indices_path"] = indices_path
        train_config["seed"] = base_seed + ref_id
        train_config["run_tag"] = f"{config.get('run_tag', 'reference')}_{ref_id:03d}"
        model_path = refs_dir / f"reference_{ref_id:03d}.pth"
        metrics_path = refs_dir / f"reference_{ref_id:03d}_metrics.json"
        command_path = refs_dir / f"reference_{ref_id:03d}_command.txt"
        resolved_config_path = refs_dir / f"reference_{ref_id:03d}_config.json"
        train_config["save_model_path"] = str(model_path)
        train_config["metrics_output_path"] = str(metrics_path)
        fingerprint = mechanism_fingerprint({"train": train_config})

        if fingerprint["sha256"] != target_fingerprint["sha256"]:
            raise RuntimeError(f"Reference {ref_id:03d} mechanism differs from target.")

        save_json(
            resolved_config_path,
            {"train": train_config, "mechanism_fingerprint": fingerprint},
        )
        command = build_main_command(
            config["python_exe"],
            config["project_root"],
            train_config,
        )
        command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

        if model_path.exists() and metrics_path.exists():
            assert_same_mechanism({"train": train_config}, _checkpoint_args(model_path))
            print(f"[OfflineLiRA] Validated reference {ref_id:03d}; skipping.")
        elif model_path.exists() or metrics_path.exists():
            raise RuntimeError(f"Reference {ref_id:03d} has incomplete artifacts.")
        else:
            print(f"[OfflineLiRA] Running reference {ref_id + 1}/{len(reference_paths)}")
            print(" ".join(command))
            subprocess.run(command, cwd=config["project_root"], check=True)

            if not model_path.is_file() or not metrics_path.is_file():
                raise RuntimeError(
                    f"Reference {ref_id:03d} completed without required artifacts."
                )

            assert_same_mechanism({"train": train_config}, _checkpoint_args(model_path))

        manifests.append(
            {
                "reference_id": ref_id,
                "indices_path": indices_path,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "command_path": str(command_path),
                "resolved_config_path": str(resolved_config_path),
                "model_sha256": sha256_file(model_path),
                "metrics_sha256": sha256_file(metrics_path),
                "mechanism_fingerprint": fingerprint,
            }
        )

    manifest = {
        "num_references": len(manifests),
        "mechanism_fingerprint": target_fingerprint,
        "references": manifests,
    }
    save_json(output_dir / "references_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Offline LiRA reference models.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_references(args.config)


if __name__ == "__main__":
    main()
