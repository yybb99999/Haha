from __future__ import annotations

import argparse

from attacks.common.config import build_main_command as build_legacy_main_command
from attacks.lira import train_target as legacy_target


LEGACY_MODULE = "attacks.lira.train_paper_stepmix"
PLD_MODULE = "attacks.lira.train_paper_stepmix_coin_aware_pld"


def build_coin_aware_pld_command(
    python_exe: str,
    project_root,
    train_config: dict,
) -> list[str]:
    if train_config.get("accountant") != "coin_aware_pld":
        raise ValueError("PLD target entrypoint requires accountant='coin_aware_pld'.")
    legacy_config = dict(train_config)
    legacy_config["accountant"] = "projected_gmm_pld"
    command = build_legacy_main_command(python_exe, project_root, legacy_config)
    module_matches = [
        index for index, value in enumerate(command) if value == LEGACY_MODULE
    ]
    if module_matches != [3]:
        raise RuntimeError(f"Unexpected legacy trainer command: {command}")
    command[3] = PLD_MODULE
    accountant_flag = command.index("--accountant")
    command[accountant_flag + 1] = "coin_aware_pld"
    return command


def train_target(config_path: str, log_path: str | None = None) -> dict:
    previous_builder = legacy_target.build_main_command
    legacy_target.build_main_command = build_coin_aware_pld_command
    try:
        return legacy_target.train_target(config_path, log_path)
    finally:
        legacy_target.build_main_command = previous_builder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a public-coin PLD StepMix target model."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    train_target(args.config, args.log_path)


if __name__ == "__main__":
    main()
