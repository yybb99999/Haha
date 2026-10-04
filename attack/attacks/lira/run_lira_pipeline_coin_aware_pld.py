from __future__ import annotations

import argparse
import subprocess

from attacks.common.config import load_json


def run_stage(config_path: str, module_name: str) -> None:
    config = load_json(config_path)
    command = [
        config["python_exe"],
        "-u",
        "-m",
        module_name,
        "--config",
        config_path,
    ]
    print("[OfflineLiRA-PLD] " + " ".join(command))
    subprocess.run(command, cwd=config["project_root"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected public-coin PLD Offline LiRA stages."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stages",
        default="target,references,export_logits,evaluate",
        help="Comma-separated stages: target,references,export_logits,evaluate.",
    )
    args = parser.parse_args()
    stage_to_module = {
        "target": "attacks.lira.train_target_coin_aware_pld",
        "references": "attacks.lira.train_references_coin_aware_pld",
        "export_logits": "attacks.lira.export_all_logits",
        "evaluate": "attacks.lira.evaluate_coin_aware_pld",
    }
    for stage in [value.strip() for value in args.stages.split(",") if value.strip()]:
        if stage not in stage_to_module:
            raise ValueError(f"Unknown stage: {stage}")
        run_stage(args.config, stage_to_module[stage])


if __name__ == "__main__":
    main()
