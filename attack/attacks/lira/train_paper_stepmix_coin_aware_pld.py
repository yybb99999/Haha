from __future__ import annotations

import argparse
import math
import os

from attacks.lira import train_paper_stepmix as legacy_stepmix
from privacy_analysis.PLD.CoinAwareStepMixPLD import (
    DP_ACCOUNTING_VERSION,
    compute_coin_aware_stepmix_pld,
)


ACCOUNTANT = "coin_aware_pld"
PLD_BACKEND = "google_dp_accounting_public_coin"
_LEGACY_VALIDATE_ARGS = legacy_stepmix._validate_args


def _validate_args(args: argparse.Namespace) -> None:
    if args.accountant != ACCOUNTANT:
        raise ValueError(f"This entrypoint requires accountant='{ACCOUNTANT}'.")
    if args.pld_backend != PLD_BACKEND:
        raise ValueError(f"This entrypoint requires pld_backend='{PLD_BACKEND}'.")
    if args.pld_backend_version != DP_ACCOUNTING_VERSION:
        raise ValueError(
            f"This entrypoint requires pld_backend_version='{DP_ACCOUNTING_VERSION}'."
        )
    if args.pld_value_discretization_interval <= 0.0:
        raise ValueError("pld_value_discretization_interval must be positive.")
    if args.pld_log_mass_truncation_bound >= 0.0:
        raise ValueError("pld_log_mass_truncation_bound must be negative.")
    if not 0.0 <= args.pld_tail_mass_truncation < 1.0:
        raise ValueError("pld_tail_mass_truncation must be in [0, 1).")
    if args.pld_max_single_step_mass_error <= 0.0:
        raise ValueError("pld_max_single_step_mass_error must be positive.")
    if args.pld_max_composed_mass_error <= 0.0:
        raise ValueError("pld_max_composed_mass_error must be positive.")
    if args.prevalidated_pld_epsilon is not None:
        if not math.isfinite(args.prevalidated_pld_epsilon):
            raise ValueError("prevalidated_pld_epsilon must be finite.")
        if args.prevalidated_pld_epsilon > args.epsilon + 1e-10:
            raise ValueError("prevalidated_pld_epsilon exceeds the epsilon budget.")
    if args.prevalidated_delta is not None and not args.privacy_certificate_sha256:
        raise ValueError(
            "Formal prevalidated PLD runs require privacy_certificate_sha256."
        )

    requested_accountant = args.accountant
    args.accountant = "projected_gmm_pld"
    try:
        _LEGACY_VALIDATE_ARGS(args)
    finally:
        args.accountant = requested_accountant


def _privacy_certificate(
    args: argparse.Namespace,
    *,
    sample_rate: float,
) -> dict:
    if args.prevalidated_delta is not None:
        return {
            "projected_delta": float(args.prevalidated_delta),
            "achieved_epsilon": args.prevalidated_pld_epsilon,
            "optimal_order": None,
            "status": "prevalidated_coin_aware_pld_v3",
        }
    if args.skip_pld_check:
        return {
            "projected_delta": None,
            "achieved_epsilon": None,
            "optimal_order": None,
            "status": "unchecked_coin_aware_pld_smoke_only",
        }

    result = compute_coin_aware_stepmix_pld(
        sample_rate=sample_rate,
        steps=args.target_steps,
        epsilon=args.epsilon,
        target_delta=args.delta,
        sigma_small=args.sigma_t,
        sigma_large=args.sigma_large,
        p_large=args.p_large,
        value_discretization_interval=args.pld_value_discretization_interval,
        log_mass_truncation_bound=args.pld_log_mass_truncation_bound,
        tail_mass_truncation=args.pld_tail_mass_truncation,
    )
    if result["projected_delta"] > args.delta:
        raise RuntimeError(
            "StepMix PLD budget exceeded: "
            f"{result['projected_delta']:.12g} > {args.delta:.12g}."
        )
    return {
        "projected_delta": float(result["projected_delta"]),
        "achieved_epsilon": float(result["achieved_epsilon"]),
        "optimal_order": None,
        "status": "computed_coin_aware_pld_v2",
    }


def train(args: argparse.Namespace) -> dict:
    previous_validator = legacy_stepmix._validate_args
    previous_certificate = legacy_stepmix._privacy_certificate
    legacy_stepmix._validate_args = _validate_args
    legacy_stepmix._privacy_certificate = _privacy_certificate
    try:
        return legacy_stepmix.train(args)
    finally:
        legacy_stepmix._validate_args = previous_validator
        legacy_stepmix._privacy_certificate = previous_certificate


def build_parser() -> argparse.ArgumentParser:
    parser = legacy_stepmix.build_parser()
    parser.description = "Paper-aligned StepMix with public-coin PLD accounting."
    parser.set_defaults(accountant=ACCOUNTANT)
    parser.add_argument("--pld_backend", default=PLD_BACKEND)
    parser.add_argument("--pld_backend_version", default=DP_ACCOUNTING_VERSION)
    parser.add_argument(
        "--pld_value_discretization_interval",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--pld_log_mass_truncation_bound",
        type=float,
        default=-50.0,
    )
    parser.add_argument("--pld_tail_mass_truncation", type=float, default=1e-15)
    parser.add_argument(
        "--pld_max_single_step_mass_error",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--pld_max_composed_mass_error",
        type=float,
        default=5e-4,
    )
    parser.add_argument("--prevalidated_pld_epsilon", type=float, default=None)
    parser.add_argument("--privacy_certificate_sha256", default=None)
    return parser


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
