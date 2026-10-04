from __future__ import annotations

import math

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as rdp_analysis


def _log_weight(weight: float) -> float:
    return -math.inf if weight == 0.0 else math.log(weight)


def coin_aware_stepmix_rdp(
    *,
    sample_rate: float,
    sigma_small: float,
    sigma_large: float,
    p_large: float,
    orders=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a per-step public-coin RDP upper bound for StepMix."""

    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must be in (0, 1].")
    if sigma_small <= 0.0:
        raise ValueError("sigma_small must be positive.")
    if sigma_large < sigma_small:
        raise ValueError("sigma_large must be at least sigma_small.")
    if not 0.0 <= p_large <= 1.0:
        raise ValueError("p_large must be in [0, 1].")

    if orders is None:
        orders = RDPAccountant.DEFAULT_ALPHAS
    orders = np.asarray(orders, dtype=np.float64)
    if np.any(orders <= 1.0):
        raise ValueError("All Renyi orders must be greater than 1.")

    small_rdp = np.asarray(
        rdp_analysis.compute_rdp(
            q=sample_rate,
            noise_multiplier=sigma_small,
            steps=1,
            orders=orders,
        ),
        dtype=np.float64,
    )
    large_rdp = np.asarray(
        rdp_analysis.compute_rdp(
            q=sample_rate,
            noise_multiplier=sigma_large,
            steps=1,
            orders=orders,
        ),
        dtype=np.float64,
    )
    alpha_minus_one = orders - 1.0
    log_small = _log_weight(1.0 - p_large) + alpha_minus_one * small_rdp
    log_large = _log_weight(p_large) + alpha_minus_one * large_rdp
    mixture_rdp = np.logaddexp(log_small, log_large) / alpha_minus_one
    return orders, mixture_rdp


def epsilon_for_coin_aware_stepmix(
    *,
    sample_rate: float,
    steps: int,
    delta: float,
    sigma_small: float,
    sigma_large: float,
    p_large: float,
    orders=None,
) -> tuple[float, float]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")

    orders, per_step_rdp = coin_aware_stepmix_rdp(
        sample_rate=sample_rate,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        p_large=p_large,
        orders=orders,
    )
    epsilon, optimal_order = rdp_analysis.get_privacy_spent(
        orders=orders,
        rdp=steps * per_step_rdp,
        delta=delta,
    )
    return float(epsilon), float(optimal_order)


def solve_sigma_small_coin_aware_rdp(
    *,
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    sigma_large: float,
    p_large: float,
    sigma_min: float = 0.1,
    tolerance: float = 1e-10,
) -> dict:
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive.")
    if sigma_min <= 0.0 or sigma_min > sigma_large:
        raise ValueError("sigma_min must be positive and no larger than sigma_large.")

    epsilon_at_large, _ = epsilon_for_coin_aware_stepmix(
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
        sigma_small=sigma_large,
        sigma_large=sigma_large,
        p_large=p_large,
    )
    if epsilon_at_large > target_epsilon:
        raise ValueError(
            "sigma_large is insufficient to bracket the requested epsilon target."
        )

    lower = sigma_min
    upper = sigma_large
    epsilon_at_min, order_at_min = epsilon_for_coin_aware_stepmix(
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
        sigma_small=lower,
        sigma_large=sigma_large,
        p_large=p_large,
    )
    if epsilon_at_min <= target_epsilon:
        return {
            "sigma_small": lower,
            "achieved_epsilon": epsilon_at_min,
            "optimal_order": order_at_min,
        }

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        epsilon, _ = epsilon_for_coin_aware_stepmix(
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
            sigma_small=midpoint,
            sigma_large=sigma_large,
            p_large=p_large,
        )
        if epsilon > target_epsilon:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break

    achieved_epsilon, optimal_order = epsilon_for_coin_aware_stepmix(
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
        sigma_small=upper,
        sigma_large=sigma_large,
        p_large=p_large,
    )
    return {
        "sigma_small": upper,
        "achieved_epsilon": achieved_epsilon,
        "optimal_order": optimal_order,
    }
