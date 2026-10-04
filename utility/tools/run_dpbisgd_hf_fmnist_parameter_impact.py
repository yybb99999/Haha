"""FMNIST HF wrapper for the audited DP-BiSGD impact-search runner.

The base search, training implementation, and privacy accountant remain
unchanged. This wrapper only registers the Fashion-MNIST handcrafted-feature
profile and appends the fixed scattering and GroupNorm options.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HF_VERSION = "fmnist_dpbisgdhf_parameter_impact_v2_cli_names"
BASE_RUNNER = Path(__file__).with_name("run_dpbisgd_parameter_impact.py")


def _load_base_runner():
    if not BASE_RUNNER.is_file():
        raise FileNotFoundError(f"Missing audited base runner: {BASE_RUNNER}")
    module_name = "_audited_dpbisgd_parameter_impact_fmnist_hf"
    spec = importlib.util.spec_from_file_location(module_name, BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_runner()

base.PROFILES["fmnist-dpbisgd-hf"] = {
    "algorithm": "DP-BiSGD-HF",
    "dataset_name": "FMNIST",
    "dataset_size": 60_000,
    "batch_size": 2_048,
    "C_t": 0.1,
    "lr": 4.0,
    "momentum": 0.9,
    "use_scattering": True,
    "input_norm": "GroupNorm",
    "num_groups": 27,
}
base.PROFILE_ALIASES["fmnist-dpsgdhf"] = "fmnist-dpbisgd-hf"


class HFImpactSearch(base.ImpactSearch):
    """Add the fixed HF feature pipeline to every candidate command."""

    def protocol(self):
        protocol = dict(super().protocol())
        protocol.update(
            {
                "version": HF_VERSION,
                "method": "DP-BiSGD-HF",
                "feature_pipeline": {
                    "use_scattering": True,
                    "input_norm": "GroupNorm",
                    "num_groups": 27,
                    "feature_transform_updated_during_training": False,
                },
                "hf_wrapper": str(Path(__file__).resolve()),
                "hf_wrapper_sha256": base._sha256(Path(__file__).resolve()),
            }
        )
        return protocol

    def command(self, candidate, plan):
        command = list(super().command(candidate, plan))
        command.extend(
            [
                "--use_scattering",
                "--input_norm",
                "GroupNorm",
                "--num_groups",
                "27",
            ]
        )
        return command

    def _csv_row(self, stage, candidate, selected, plan, metrics):
        row = dict(super()._csv_row(stage, candidate, selected, plan, metrics))
        row["Method"] = "DP-BiSGD-HF"
        return row


base.VERSION = HF_VERSION
base.ImpactSearch = HFImpactSearch


if __name__ == "__main__":
    base.main()
