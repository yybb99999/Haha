from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


DP_ACCOUNTING_DISTRIBUTION = "dp-accounting"
DP_ACCOUNTING_VERSION = "0.5.1"


def _load_backend():
    try:
        installed_version = importlib.metadata.version(DP_ACCOUNTING_DISTRIBUTION)
        from dp_accounting.pld import pld_pmf
        from dp_accounting.pld import privacy_loss_distribution as pld
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "coin_aware_pld requires the isolated dp-accounting==0.5.1 backend."
        ) from exc

    if installed_version != DP_ACCOUNTING_VERSION:
        raise RuntimeError(
            "coin_aware_pld requires dp-accounting=="
            f"{DP_ACCOUNTING_VERSION}, found {installed_version}."
        )
    return pld, pld_pmf, installed_version


def _dense_public_coin_mixture(
    small_pmf: Any,
    large_pmf: Any,
    *,
    p_large: float,
    pld_pmf_module: Any,
):
    """Mixes branch PLDs for a public, data-independent branch label."""
    small = small_pmf.to_dense_pmf()
    large = large_pmf.to_dense_pmf()
    required = (
        "_discretization",
        "_infinity_mass",
        "_lower_loss",
        "_pessimistic_estimate",
        "_probs",
    )
    if any(not hasattr(small, name) for name in required) or any(
        not hasattr(large, name) for name in required
    ):
        raise RuntimeError("Unsupported dp-accounting PLD PMF representation.")
    if small._discretization != large._discretization:
        raise ValueError("StepMix branch PLDs use different discretizations.")
    if not small._pessimistic_estimate or not large._pessimistic_estimate:
        raise ValueError("Formal StepMix PLD accounting must be pessimistic.")

    lower_loss = min(small._lower_loss, large._lower_loss)
    upper_loss = max(
        small._lower_loss + small.size,
        large._lower_loss + large.size,
    )
    probabilities = np.zeros(upper_loss - lower_loss, dtype=np.float64)
    small_offset = small._lower_loss - lower_loss
    large_offset = large._lower_loss - lower_loss
    probabilities[small_offset : small_offset + small.size] += (
        1.0 - p_large
    ) * small._probs
    probabilities[large_offset : large_offset + large.size] += (
        p_large * large._probs
    )
    infinity_mass = (
        (1.0 - p_large) * small._infinity_mass
        + p_large * large._infinity_mass
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError("Public-coin PLD mixture contains invalid probability mass.")
    if not math.isfinite(infinity_mass) or not 0.0 <= infinity_mass <= 1.0:
        raise RuntimeError("Public-coin PLD mixture has invalid infinity mass.")

    return pld_pmf_module.DensePLDPmf(
        small._discretization,
        lower_loss,
        probabilities,
        infinity_mass,
        True,
    )


def _pmf_diagnostics(pmf: Any) -> dict[str, float | int]:
    dense = pmf.to_dense_pmf()
    upper_loss = dense._lower_loss + dense.size - 1
    finite_mass = float(np.sum(dense._probs))
    infinity_mass = float(dense._infinity_mass)
    total_mass = finite_mass + infinity_mass
    return {
        "size": int(dense.size),
        "lower_loss": float(dense._lower_loss * dense._discretization),
        "upper_loss": float(upper_loss * dense._discretization),
        "finite_mass": finite_mass,
        "infinity_mass": infinity_mass,
        "total_mass": total_mass,
        "absolute_mass_error": abs(total_mass - 1.0),
    }


@dataclass(frozen=True)
class CoinAwareStepMixPLDAccountant:
    """Pessimistic PLD accountant for the public-coin StepMix mechanism.

    The branch label is data-independent and public. Conditional on that label,
    one update is a Poisson-sampled Gaussian mechanism. Its exact one-step PLD
    is therefore the probability-weighted mixture of the two branch PLDs.
    """

    sample_rate: float
    sigma_small: float
    sigma_large: float
    p_large: float
    value_discretization_interval: float = 1e-4
    log_mass_truncation_bound: float = -50.0
    tail_mass_truncation: float = 1e-15

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1].")
        if self.sigma_small <= 0.0:
            raise ValueError("sigma_small must be positive.")
        if self.sigma_large < self.sigma_small:
            raise ValueError("sigma_large must be at least sigma_small.")
        if not 0.0 <= self.p_large <= 1.0:
            raise ValueError("p_large must be in [0, 1].")
        if self.value_discretization_interval <= 0.0:
            raise ValueError("value_discretization_interval must be positive.")
        if self.log_mass_truncation_bound >= 0.0:
            raise ValueError("log_mass_truncation_bound must be negative.")
        if not 0.0 <= self.tail_mass_truncation < 1.0:
            raise ValueError("tail_mass_truncation must be in [0, 1).")

    def _branch_pld(self, sigma: float):
        pld, _, _ = _load_backend()
        return pld.from_gaussian_mechanism(
            standard_deviation=float(sigma),
            sensitivity=1.0,
            pessimistic_estimate=True,
            value_discretization_interval=self.value_discretization_interval,
            log_mass_truncation_bound=self.log_mass_truncation_bound,
            sampling_prob=self.sample_rate,
            use_connect_dots=True,
        )

    def _single_step_pld(self):
        pld, pld_pmf, _ = _load_backend()
        small = self._branch_pld(self.sigma_small)
        large = self._branch_pld(self.sigma_large)
        remove = _dense_public_coin_mixture(
            small._pmf_remove,
            large._pmf_remove,
            p_large=self.p_large,
            pld_pmf_module=pld_pmf,
        )
        add = _dense_public_coin_mixture(
            small._pmf_add,
            large._pmf_add,
            p_large=self.p_large,
            pld_pmf_module=pld_pmf,
        )
        return pld.PrivacyLossDistribution(remove, add)

    def compute(
        self,
        *,
        steps: int,
        epsilon: float,
        target_delta: float | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative.")
        if target_delta is not None and not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1).")

        _, _, backend_version = _load_backend()
        single_step = self._single_step_pld()
        composed = single_step.self_compose(
            steps,
            tail_mass_truncation=self.tail_mass_truncation,
        )
        projected_delta = float(composed.get_delta_for_epsilon(epsilon))
        achieved_epsilon = (
            float(composed.get_epsilon_for_delta(target_delta))
            if target_delta is not None
            else None
        )
        if not math.isfinite(projected_delta):
            raise RuntimeError("PLD backend returned a non-finite delta.")
        if achieved_epsilon is not None and not math.isfinite(achieved_epsilon):
            raise RuntimeError("PLD backend returned a non-finite epsilon.")

        result: dict[str, Any] = {
            "projected_delta": projected_delta,
            "achieved_delta": projected_delta,
            "achieved_epsilon": achieved_epsilon,
            "backend_distribution": DP_ACCOUNTING_DISTRIBUTION,
            "backend_version": backend_version,
            "pessimistic_estimate": True,
            "use_connect_dots": True,
        }
        if return_diagnostics:
            result["diagnostics"] = {
                "single_step_remove": _pmf_diagnostics(single_step._pmf_remove),
                "single_step_add": _pmf_diagnostics(single_step._pmf_add),
                "composed_remove": _pmf_diagnostics(composed._pmf_remove),
                "composed_add": _pmf_diagnostics(composed._pmf_add),
            }
        return result


def compute_coin_aware_stepmix_pld(
    *,
    sample_rate: float,
    steps: int,
    epsilon: float,
    sigma_small: float,
    sigma_large: float,
    p_large: float,
    target_delta: float | None = None,
    value_discretization_interval: float = 1e-4,
    log_mass_truncation_bound: float = -50.0,
    tail_mass_truncation: float = 1e-15,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    accountant = CoinAwareStepMixPLDAccountant(
        sample_rate=sample_rate,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        p_large=p_large,
        value_discretization_interval=value_discretization_interval,
        log_mass_truncation_bound=log_mass_truncation_bound,
        tail_mass_truncation=tail_mass_truncation,
    )
    return accountant.compute(
        steps=steps,
        epsilon=epsilon,
        target_delta=target_delta,
        return_diagnostics=return_diagnostics,
    )


def solve_sigma_small_coin_aware_pld(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    sigma_large: float,
    p_large: float,
    sigma_min: float = 0.1,
    sigma_max: float | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 80,
    value_discretization_interval: float = 1e-4,
    log_mass_truncation_bound: float = -50.0,
    tail_mass_truncation: float = 1e-15,
) -> dict[str, Any]:
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive.")
    if not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must be in (0, 1).")
    if sigma_min <= 0.0 or tolerance <= 0.0:
        raise ValueError("sigma_min and tolerance must be positive.")

    def evaluate(sigma_small: float, *, diagnostics: bool = False) -> dict[str, Any]:
        return compute_coin_aware_stepmix_pld(
            sample_rate=sample_rate,
            steps=steps,
            epsilon=target_epsilon,
            target_delta=target_delta,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            p_large=p_large,
            value_discretization_interval=value_discretization_interval,
            log_mass_truncation_bound=log_mass_truncation_bound,
            tail_mass_truncation=tail_mass_truncation,
            return_diagnostics=diagnostics,
        )

    high = float(sigma_max) if sigma_max is not None else min(1.0, sigma_large)
    if high < sigma_min or high > sigma_large:
        raise ValueError("sigma_max must be in [sigma_min, sigma_large].")
    high_result = evaluate(high)
    while high_result["projected_delta"] > target_delta and high < sigma_large:
        high = min(2.0 * high, sigma_large)
        high_result = evaluate(high)
    if high_result["projected_delta"] > target_delta:
        raise ValueError("sigma_large does not satisfy the requested PLD budget.")

    low = max(sigma_min, high / 2.0)
    low_result = evaluate(low)
    while low > sigma_min and low_result["projected_delta"] <= target_delta:
        high = low
        high_result = low_result
        low = max(sigma_min, low / 2.0)
        low_result = evaluate(low)
    if low == sigma_min and low_result["projected_delta"] <= target_delta:
        final = evaluate(low, diagnostics=True)
        return {
            "sigma_small": low,
            "projected_delta": final["projected_delta"],
            "achieved_delta": final["achieved_delta"],
            "achieved_epsilon": final["achieved_epsilon"],
            "infeasible_sigma": None,
            "infeasible_delta": None,
            "iterations": 0,
            **{key: final[key] for key in final if key not in {
                "projected_delta", "achieved_delta", "achieved_epsilon"
            }},
        }

    iterations = 0
    while high - low > tolerance:
        if iterations >= max_iterations:
            raise RuntimeError("PLD sigma calibration did not converge.")
        midpoint = (low + high) / 2.0
        midpoint_result = evaluate(midpoint)
        if midpoint_result["projected_delta"] <= target_delta:
            high = midpoint
            high_result = midpoint_result
        else:
            low = midpoint
            low_result = midpoint_result
        iterations += 1

    final = evaluate(high, diagnostics=True)
    if final["projected_delta"] > target_delta:
        raise RuntimeError("Calibrated PLD sigma does not satisfy target_delta.")
    return {
        "sigma_small": float(high),
        "projected_delta": final["projected_delta"],
        "achieved_delta": final["achieved_delta"],
        "achieved_epsilon": final["achieved_epsilon"],
        "infeasible_sigma": float(low),
        "infeasible_delta": float(low_result["projected_delta"]),
        "iterations": int(iterations),
        **{key: final[key] for key in final if key not in {
            "projected_delta", "achieved_delta", "achieved_epsilon"
        }},
    }
