from __future__ import annotations

import argparse

from attacks.lira import train_references as legacy_references
from attacks.lira.train_target_coin_aware_pld import (
    build_coin_aware_pld_command,
)


def train_references(config_path: str) -> dict:
    previous_builder = legacy_references.build_main_command
    legacy_references.build_main_command = build_coin_aware_pld_command
    try:
        return legacy_references.train_references(config_path)
    finally:
        legacy_references.build_main_command = previous_builder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train public-coin PLD StepMix Offline LiRA references."
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_references(args.config)


if __name__ == "__main__":
    main()
