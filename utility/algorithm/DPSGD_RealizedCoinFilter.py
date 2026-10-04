"""Ordinary DP-BiSGD with a realized-coin PLD privacy filter."""

import torch

from algorithm.realized_coin_filter_training import (
    train_with_realized_coin_filter,
)


def DPSGD_RealizedCoinFilter(
    train_data,
    test_data,
    model,
    optimizer,
    batch_size,
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
    trace_callback=None,
    run_context=None,
):
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_with_realized_coin_filter(
        train_data=train_data,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        expected_batch_size=batch_size,
        epsilon_budget=epsilon_budget,
        delta=delta,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        p_large=p_large,
        device=device,
        max_updates_safety=max_updates_safety,
        branch_seed=branch_seed,
        poisson_private_seed=poisson_private_seed,
        pld_discretization=pld_discretization,
        pld_log_mass_truncation=pld_log_mass_truncation,
        pld_tail_mass_truncation=pld_tail_mass_truncation,
        trace_callback=trace_callback,
        run_context=run_context,
    )
