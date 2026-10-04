"""Shared training loop for privacy-filtered DP-BiSGD variants."""

from typing import Callable, Dict, Optional

import torch

from privacy_analysis.filters import (
    PessimisticRealizedCoinPLDFilter,
    RealizedCoinRDPFilter,
)
from train_and_validation.train_with_dp import train_with_dp
from train_and_validation.validation import validation
from utils.realized_coin_sampling import PoissonStepSampler, RealizedCoinSampler


def _single_batch_loader(dataset, indices):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=[indices.tolist()],
        num_workers=0,
        pin_memory=False,
    )


def train_with_realized_coin_filter(
    train_data,
    test_loader,
    model,
    optimizer,
    expected_batch_size,
    epsilon_budget,
    delta,
    sigma_small,
    sigma_large,
    p_large,
    device,
    max_updates_safety,
    branch_seed,
    poisson_private_seed=None,
    pld_discretization=1e-4,
    pld_log_mass_truncation=-50.0,
    pld_tail_mass_truncation=1e-15,
    trace_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    run_context=None,
):
    """Train until the next realized branch would exceed the PLD budget."""

    if max_updates_safety <= 0:
        raise ValueError("max_updates_safety must be positive.")
    if optimizer.dp_update_steps != 0:
        raise ValueError("The optimizer must be unused at training start.")
    if int(optimizer.minibatch_size) != int(expected_batch_size):
        raise ValueError(
            "The optimizer denominator must equal the expected Poisson batch size."
        )

    sample_rate = expected_batch_size / len(train_data)
    pld_filter = PessimisticRealizedCoinPLDFilter(
        target_epsilon=epsilon_budget,
        target_delta=delta,
        sample_rate=sample_rate,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        value_discretization_interval=pld_discretization,
        log_mass_truncation_bound=pld_log_mass_truncation,
        tail_mass_truncation=pld_tail_mass_truncation,
    )
    rdp_filter = RealizedCoinRDPFilter(
        sample_rate=sample_rate,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        target_delta=delta,
    )
    branch_sampler = RealizedCoinSampler(
        p_large=p_large,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        seed=branch_seed,
    )
    poisson_sampler = PoissonStepSampler(
        dataset_size=len(train_data),
        expected_batch_size=expected_batch_size,
        private_seed=poisson_private_seed,
    )

    def emit(event):
        if trace_callback is not None:
            trace_callback(event)

    initial_loss, initial_acc = validation(model, test_loader, device)
    test_accuracy = float(initial_acc)
    test_loss = float(initial_loss)
    best_test_acc = test_accuracy
    best_iter = 0
    epsilon_list = []
    test_loss_list = []
    stop_reason = None
    rejected_decision = None
    rejected_rdp = None

    while pld_filter.committed_steps < max_updates_safety:
        coin = branch_sampler.draw()
        pld_decision = pld_filter.preview(coin.branch)
        rdp_decision = rdp_filter.preview(
            coin.branch,
            token=pld_decision.token,
        )

        if not pld_decision.allowed:
            rdp_filter.cancel(pld_decision.token)
            rejected_decision = pld_decision
            rejected_rdp = rdp_decision
            stop_reason = "privacy_budget"
            emit(
                {
                    "event": "privacy_stop",
                    "coin": coin.to_dict(),
                    "pld": pld_decision.to_dict(),
                    "rdp_epsilon_upper": rdp_decision.epsilon_upper,
                    "rdp_best_order": rdp_decision.best_order,
                    "committed_steps": pld_filter.committed_steps,
                }
            )
            break

        indices = poisson_sampler.draw_indices()
        actual_batch_size = int(indices.numel())
        optimizer.authorize_step(
            token=pld_decision.token,
            branch=coin.branch,
            sigma=coin.sigma,
        )
        updates_before = optimizer.dp_update_steps

        if actual_batch_size == 0:
            optimizer.zero_accum_grad()
            optimizer.step_dp()
        else:
            train_loader = _single_batch_loader(train_data, indices)
            train_with_dp(model, train_loader, optimizer, device)

        if optimizer.dp_update_steps != updates_before + 1:
            raise RuntimeError("A private round must execute exactly one update.")

        pld_filter.commit(pld_decision.token)
        rdp_filter.commit(pld_decision.token)
        optimizer.confirm_completed_step(pld_decision.token)

        accounted_steps = pld_filter.committed_steps
        if accounted_steps != optimizer.dp_update_steps:
            raise RuntimeError("Accountant and optimizer update counts diverged.")
        if accounted_steps != rdp_filter.committed_steps:
            raise RuntimeError("PLD and RDP accounting counts diverged.")

        test_loss_value, test_accuracy_value = validation(
            model,
            test_loader,
            device,
        )
        test_loss = float(test_loss_value)
        test_accuracy = float(test_accuracy_value)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = accounted_steps

        epsilon_list.append(pld_filter.current_epsilon_upper())
        test_loss_list.append(test_loss)
        emit(
            {
                "event": "committed_update",
                "step": accounted_steps,
                "coin": coin.to_dict(),
                "normalization_batch_size": int(expected_batch_size),
                "pld_delta_upper": pld_filter.current_delta_upper(),
                "pld_epsilon_upper": pld_filter.current_epsilon_upper(),
                "rdp_epsilon_upper": rdp_filter.current_epsilon()[0],
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_test_acc,
                "best_iter": best_iter,
            }
        )
        print(
            f"iters:{accounted_steps}, accountant:realized_coin_pld_filter, "
            f"epsilon_upper:{pld_filter.current_epsilon_upper():.6f}, "
            f"delta_upper@target_eps:{pld_filter.current_delta_upper():.3e}, "
            f"branch:{coin.branch}, sigma:{coin.sigma:.6f}, "
            f"expected_batch:{expected_batch_size} | "
            f"Test set: Average loss:{test_loss:.4f}, "
            f"Accuracy:({test_accuracy:.2f}%)"
        )

    if stop_reason is None:
        stop_reason = "max_updates_safety"

    accounted_steps = pld_filter.committed_steps
    if accounted_steps != optimizer.dp_update_steps:
        raise RuntimeError("Final accountant and optimizer counts diverged.")
    if accounted_steps != (
        optimizer.stepwise_small_steps + optimizer.stepwise_large_steps
    ):
        raise RuntimeError("Final StepMix branch counts are inconsistent.")

    observed_p_large = (
        optimizer.stepwise_large_steps / accounted_steps
        if accounted_steps > 0
        else None
    )
    status = "passed" if stop_reason == "privacy_budget" else "incomplete"
    pld_state = pld_filter.state_dict()
    rdp_state = rdp_filter.state_dict()
    metrics = {
        "status": status,
        "stop_reason": stop_reason,
        "final_acc": test_accuracy,
        "final_iter": accounted_steps,
        "T_final": accounted_steps,
        "best_acc": best_test_acc,
        "best_iter": best_iter,
        "initial_acc": float(initial_acc),
        "initial_loss": float(initial_loss),
        "epsilon_target": float(epsilon_budget),
        "delta_target": float(delta),
        "epsilon_upper_final": pld_filter.current_epsilon_upper(),
        "delta_upper_final_at_target_epsilon": (
            pld_filter.current_delta_upper()
        ),
        "accountant": "pessimistic_realized_public_coin_pld_filter",
        "privacy_status": status,
        "privacy_filter_order": "draw_preview_sample_update_commit",
        "privacy_guarantee_scope": "realized_public_coin_pathwise",
        "stop_before_exceed": True,
        "sampling": "poisson_independent",
        "sample_rate": sample_rate,
        "batch_size": int(expected_batch_size),
        "normalization": "expected_batch_size",
        "microbatch_size": 1,
        "sigma_small": float(sigma_small),
        "sigma_large": float(sigma_large),
        "p_large": float(p_large),
        "accounted_steps": accounted_steps,
        "actual_dp_updates": int(optimizer.dp_update_steps),
        "stepwise_total_steps": int(optimizer.stepwise_total_steps),
        "stepwise_small_steps": int(optimizer.stepwise_small_steps),
        "stepwise_large_steps": int(optimizer.stepwise_large_steps),
        "observed_p_large": observed_p_large,
        "branch_draws": int(branch_sampler.draw_count),
        "poisson_draws": int(poisson_sampler.draw_count),
        "branch_seed": int(branch_seed),
        "poisson_private_seed_commitment": (
            poisson_sampler.private_seed_commitment
        ),
        "noise_private_seed_commitment": optimizer.noise_seed_commitment,
        "private_rng_seeds_released": False,
        "max_updates_safety": int(max_updates_safety),
        "pld_state": pld_state,
        "rdp_reference_state": rdp_state,
        "rejected_next_branch": (
            rejected_decision.branch if rejected_decision else None
        ),
        "rejected_next_sigma": (
            rejected_decision.sigma if rejected_decision else None
        ),
        "rejected_next_delta_upper": (
            rejected_decision.delta_upper if rejected_decision else None
        ),
        "rejected_next_epsilon_upper": (
            rejected_decision.epsilon_upper if rejected_decision else None
        ),
        "rejected_next_rdp_epsilon_upper": (
            rejected_rdp.epsilon_upper if rejected_rdp else None
        ),
        "Best_Acc": best_test_acc,
        "Fin_Acc": test_accuracy,
        "Sigma_Small": float(sigma_small),
        "Sigma_Large": float(sigma_large),
        "P_Large": float(p_large),
        "Best_Acc_t": best_iter,
        "Fin_Acc_T": accounted_steps,
    }
    if run_context:
        metrics.update(run_context)

    return (
        test_accuracy,
        accounted_steps,
        best_test_acc,
        best_iter,
        model,
        [epsilon_list, test_loss_list],
        metrics,
    )
