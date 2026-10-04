from __future__ import annotations

import argparse
from pathlib import Path

from attacks.common.config import load_json
from attacks.lira.export_logits import export_checkpoint_logits


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def export_all_logits(config_path: str) -> None:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    split_manifest = load_json(config["split_manifest_path"])
    candidate_indices = split_manifest["candidate_indices_path"]
    expected_k = int(split_manifest["num_references"])
    train = config["train"]
    export = config.get("export", {})
    batch_size = int(export.get("batch_size", train.get("batch_size", 1024)))
    device = export.get("device", train.get("device", "cuda"))
    candidate_pool = export.get("candidate_pool", "train")
    query_augmentations = export.get("query_augmentations", ["identity"])
    logits_subdir = export.get("logits_subdir")
    logits_dir = output_dir / logits_subdir if logits_subdir else output_dir
    logits_refs_dir = logits_dir / "references"
    logits_refs_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = _require_file(output_dir / "target_model.pth")
    refs_dir = output_dir / "references"

    export_checkpoint_logits(
        checkpoint_path=str(target_model_path),
        output_path=str(logits_dir / "target_logits.npz"),
        dataset_name=train["dataset_name"],
        algorithm=train["algorithm"],
        indices_path=candidate_indices,
        batch_size=batch_size,
        device=device,
        use_scattering=bool(train.get("use_scattering", False)),
        input_norm=train.get("input_norm"),
        num_groups=int(train.get("num_groups", 27)),
        candidate_pool=candidate_pool,
        query_augmentations=query_augmentations,
    )

    for ref_id in range(expected_k):
        model_path = _require_file(refs_dir / f"reference_{ref_id:03d}.pth")
        export_checkpoint_logits(
            checkpoint_path=str(model_path),
            output_path=str(
                logits_refs_dir / f"reference_{ref_id:03d}_logits.npz"
            ),
            dataset_name=train["dataset_name"],
            algorithm=train["algorithm"],
            indices_path=candidate_indices,
            batch_size=batch_size,
            device=device,
            use_scattering=bool(train.get("use_scattering", False)),
            input_norm=train.get("input_norm"),
            num_groups=int(train.get("num_groups", 27)),
            candidate_pool=candidate_pool,
            query_augmentations=query_augmentations,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all Offline LiRA logits.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    export_all_logits(args.config)


if __name__ == "__main__":
    main()
