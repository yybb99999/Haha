from __future__ import annotations

import argparse

from attacks.lira.train_target_coin_aware_pld import build_coin_aware_pld_command
from tools import train_lira_reference_one as legacy_reference_one


def train_reference_one(config_path: str, ref_id: int, log_path: str | None = None) -> None:
    legacy_builder = legacy_reference_one.build_main_command
    legacy_reference_one.build_main_command = build_coin_aware_pld_command
    try:
        legacy_reference_one.train_reference_one(config_path, ref_id, log_path)
    finally:
        legacy_reference_one.build_main_command = legacy_builder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Offline LiRA reference model with coin-aware PLD validation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref_id", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    train_reference_one(args.config, args.ref_id, args.log_path)


if __name__ == "__main__":
    main()
