from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from attacks.common.config import (  
    assert_same_mechanism,
    build_main_command,
    load_json,
    mechanism_fingerprint,
    save_json,
    sha256_file,
)
from attacks.common.locking import acquire_lock, release_lock  


def _checkpoint_args(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise RuntimeError(f"Invalid audit checkpoint: {path}")

    return payload["args"]


def train_reference_one(config_path: str, ref_id: int, log_path: str | None = None) -> None:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    refs_dir = output_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = load_json(config["split_manifest_path"])
    reference_paths = split_manifest["reference_train_indices_paths"]

    if ref_id < 0 or ref_id >= len(reference_paths):
        raise ValueError(f"ref_id out of range: {ref_id}")

    base_seed = int(config.get("reference_seed_base", 10000))
    model_path = refs_dir / f"reference_{ref_id:03d}.pth"
    metrics_path = refs_dir / f"reference_{ref_id:03d}_metrics.json"
    command_path = refs_dir / f"reference_{ref_id:03d}_command.txt"
    resolved_path = refs_dir / f"reference_{ref_id:03d}_config.json"
    lock_path = refs_dir / f"reference_{ref_id:03d}.lock"
    train_config = dict(config["train"])
    train_config["train_indices_path"] = reference_paths[ref_id]
    train_config["seed"] = base_seed + ref_id
    train_config["run_tag"] = f"{config.get('run_tag', 'reference')}_{ref_id:03d}"
    train_config["save_model_path"] = str(model_path)
    train_config["metrics_output_path"] = str(metrics_path)
    fingerprint = mechanism_fingerprint({"train": train_config})
    save_json(resolved_path, {"train": train_config, "mechanism_fingerprint": fingerprint})

    if model_path.exists() and metrics_path.exists():
        assert_same_mechanism({"train": train_config}, _checkpoint_args(model_path))
        print(f"[OfflineLiRA] Reference {ref_id:03d} already complete and valid.")
        return

    if model_path.exists() or metrics_path.exists():
        raise RuntimeError(f"Reference {ref_id:03d} has incomplete artifacts.")

    command = build_main_command(
        config["python_exe"],
        config["project_root"],
        train_config,
    )
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    acquire_lock(
        lock_path,
        reference_id=ref_id,
        mechanism_fingerprint=fingerprint["sha256"],
        timeout_hours=float(config.get("lock_timeout_hours", 24.0)),
        force_unlock_stale=bool(config.get("force_unlock_stale", False)),
    )

    try:
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
            raise RuntimeError("Reference completed without required artifacts.")

        assert_same_mechanism({"train": train_config}, _checkpoint_args(model_path))
        save_json(
            refs_dir / f"reference_{ref_id:03d}_artifact_manifest.json",
            {
                "reference_id": ref_id,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "model_sha256": sha256_file(model_path),
                "metrics_sha256": sha256_file(metrics_path),
                "mechanism_fingerprint": fingerprint,
            },
        )
    finally:
        release_lock(lock_path, expected_pid=os.getpid())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Offline LiRA reference model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref_id", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    train_reference_one(args.config, args.ref_id, args.log_path)


if __name__ == "__main__":
    main()
