from __future__ import annotations

import argparse
import json
from pathlib import Path

from attacks.common.config import save_json
from privacy_analysis.PLD.CoinAwareStepMixPLD import (
    DP_ACCOUNTING_DISTRIBUTION,
    DP_ACCOUNTING_VERSION,
    solve_sigma_small_coin_aware_pld,
)


def _validate_mass_diagnostics(
    diagnostics: dict,
    *,
    max_single_step_error: float,
    max_composed_error: float,
) -> dict:
    single_keys = ("single_step_remove", "single_step_add")
    composed_keys = ("composed_remove", "composed_add")
    single_error = max(
        float(diagnostics[key]["absolute_mass_error"]) for key in single_keys
    )
    composed_error = max(
        float(diagnostics[key]["absolute_mass_error"]) for key in composed_keys
    )
    if single_error > max_single_step_error:
        raise RuntimeError(
            "Single-step PLD mass error exceeds the numerical gate: "
            f"{single_error:.12g} > {max_single_step_error:.12g}."
        )
    if composed_error > max_composed_error:
        raise RuntimeError(
            "Composed PLD mass error exceeds the numerical gate: "
            f"{composed_error:.12g} > {max_composed_error:.12g}."
        )
    return {
        "status": "passed",
        "single_step_max_absolute_mass_error": single_error,
        "composed_max_absolute_mass_error": composed_error,
        "max_single_step_absolute_mass_error": max_single_step_error,
        "max_composed_absolute_mass_error": max_composed_error,
    }


def calibrate(args: argparse.Namespace) -> dict:
    if args.train_size <= 0 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("train_size, batch_size, and steps must be positive.")
    if args.batch_size > args.train_size:
        raise ValueError("batch_size cannot exceed train_size.")
    if args.C_t <= 0.0:
        raise ValueError("C_t must be positive.")
    if any(epsilon <= 0.0 for epsilon in args.target_epsilons):
        raise ValueError("All target epsilons must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = args.batch_size / float(args.train_size)
    rows = []

    for epsilon in args.target_epsilons:
        output_path = (
            output_dir
            / f"cifar10_eps{epsilon:g}_paper_stepmix_coin_aware_pld.json"
        )
        if output_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite certificate: {output_path}")

        solved = solve_sigma_small_coin_aware_pld(
            target_epsilon=epsilon,
            target_delta=args.delta,
            sample_rate=sample_rate,
            steps=args.steps,
            sigma_large=args.sigma_large,
            p_large=args.p_large,
            sigma_min=args.sigma_search_min,
            sigma_max=args.sigma_search_max,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            value_discretization_interval=args.value_discretization_interval,
            log_mass_truncation_bound=args.log_mass_truncation_bound,
            tail_mass_truncation=args.tail_mass_truncation,
        )
        if solved["projected_delta"] > args.delta:
            raise RuntimeError("Calibrated StepMix PLD certificate exceeds delta.")
        numerical_validation = _validate_mass_diagnostics(
            solved["diagnostics"],
            max_single_step_error=args.max_single_step_mass_error,
            max_composed_error=args.max_composed_mass_error,
        )

        certificate = {
            "dataset_name": "CIFAR-10",
            "epsilon": float(epsilon),
            "target_delta": float(args.delta),
            "projected_delta": float(solved["projected_delta"]),
            "achieved_delta": float(solved["achieved_delta"]),
            "achieved_epsilon": float(solved["achieved_epsilon"]),
            "train_size": int(args.train_size),
            "batch_size": int(args.batch_size),
            "sample_rate": sample_rate,
            "target_steps": int(args.steps),
            "C_t": float(args.C_t),
            "sigma_small": float(solved["sigma_small"]),
            "sigma_large": float(args.sigma_large),
            "p_large": float(args.p_large),
            "mixpld_mode": "coin_aware",
            "accountant": "coin_aware_pld",
            "public_coin": True,
            "pessimistic_estimate": True,
            "use_connect_dots": True,
            "pld_backend": "google_dp_accounting_public_coin",
            "backend_distribution": DP_ACCOUNTING_DISTRIBUTION,
            "backend_version": DP_ACCOUNTING_VERSION,
            "value_discretization_interval": float(
                args.value_discretization_interval
            ),
            "log_mass_truncation_bound": float(args.log_mass_truncation_bound),
            "tail_mass_truncation": float(args.tail_mass_truncation),
            "large_step_update": args.large_step_update,
            "large_step_lr_scale": float(args.large_step_lr_scale),
            "sigma_tolerance": float(args.tolerance),
            "infeasible_sigma": solved["infeasible_sigma"],
            "infeasible_delta": solved["infeasible_delta"],
            "solver_iterations": int(solved["iterations"]),
            "solver_version": "isolated_paper_stepmix_coin_aware_pld_v3",
            "diagnostics": solved["diagnostics"],
            "numerical_validation": numerical_validation,
            "source": (
                "Pessimistic Google dp-accounting PLDs for each Poisson-sampled "
                "Gaussian branch, mixed by the public data-independent branch "
                "probability before fixed-step self-composition."
            ),
        }
        save_json(output_path, certificate)
        rows.append({"path": str(output_path), **certificate})

    manifest = {
        "status": "passed",
        "protocol": {
            "train_size": args.train_size,
            "batch_size": args.batch_size,
            "sample_rate": sample_rate,
            "steps": args.steps,
            "delta": args.delta,
            "C_t": args.C_t,
            "sigma_large": args.sigma_large,
            "p_large": args.p_large,
            "large_step_update": args.large_step_update,
            "large_step_lr_scale": args.large_step_lr_scale,
        },
        "backend": {
            "distribution": DP_ACCOUNTING_DISTRIBUTION,
            "version": DP_ACCOUNTING_VERSION,
            "pld_backend": "google_dp_accounting_public_coin",
            "pessimistic_estimate": True,
            "use_connect_dots": True,
            "value_discretization_interval": args.value_discretization_interval,
            "log_mass_truncation_bound": args.log_mass_truncation_bound,
            "tail_mass_truncation": args.tail_mass_truncation,
            "max_single_step_mass_error": args.max_single_step_mass_error,
            "max_composed_mass_error": args.max_composed_mass_error,
        },
        "rows": rows,
        "accounting_note": (
            "The branch label is public and data-independent. The one-step PLD "
            "is the weighted mixture of branch PLDs in both add and remove "
            "directions; pessimistic discretization and tail truncation are "
            "retained through composition. Numerical PMF mass is gated before "
            "a certificate is accepted."
        ),
    }
    save_json(output_dir / "stepmix_coin_aware_pld_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate public-coin StepMix PLD for paper comparisons."
    )
    parser.add_argument("--train_size", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=9800)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--C_t", type=float, default=1.0)
    parser.add_argument("--sigma_large", type=float, default=20.0)
    parser.add_argument("--p_large", type=float, default=0.04)
    parser.add_argument("--large_step_update", default="sgd_bypass_scaled")
    parser.add_argument("--large_step_lr_scale", type=float, default=0.1)
    parser.add_argument(
        "--target_epsilons",
        type=float,
        nargs="+",
        default=[8.0, 4.0, 1.0],
    )
    parser.add_argument("--sigma_search_min", type=float, default=0.1)
    parser.add_argument("--sigma_search_max", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max_iterations", type=int, default=80)
    parser.add_argument(
        "--value_discretization_interval",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--log_mass_truncation_bound", type=float, default=-50.0)
    parser.add_argument("--tail_mass_truncation", type=float, default=1e-15)
    parser.add_argument("--max_single_step_mass_error", type=float, default=1e-6)
    parser.add_argument("--max_composed_mass_error", type=float, default=5e-4)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "certificates_recomputed",
    )
    parser.add_argument("--force", action="store_true")
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
