"""DP-BiSGD-HF with a realized-coin PLD privacy filter."""

import torch

from algorithm.realized_coin_filter_training import (
    train_with_realized_coin_filter,
)
from data.util.get_data import (
    get_scatter_transform,
    get_scattered_dataset,
    get_scattered_loader,
)
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from utils.preselected_step_dp_optimizer import (
    get_preselected_step_dpsgd_optimizer,
)


CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}


def DPSGD_HF_RealizedCoinFilter(
    dataset_name,
    train_data,
    test_data,
    batch_size,
    lr,
    momentum,
    epsilon_budget,
    delta,
    C_t,
    sigma_small,
    sigma_large,
    p_large,
    use_scattering,
    input_norm,
    num_groups,
    device,
    max_updates_safety,
    branch_seed,
    poisson_private_seed=None,
    noise_private_seed=None,
    large_step_update="sgd_bypass_scaled",
    large_step_lr_scale=0.1,
    pld_discretization=1e-4,
    pld_log_mass_truncation=-50.0,
    pld_tail_mass_truncation=1e-15,
    trace_callback=None,
    run_context=None,
):
    if dataset_name not in CNNS:
        raise ValueError(f"DP-BiSGD-HF does not support {dataset_name}.")
    if not use_scattering:
        raise ValueError("DP-BiSGD-HF requires the scattering feature transform.")
    if input_norm != "GroupNorm":
        raise ValueError(
            "DP-BiSGD-HF requires input_norm='GroupNorm'; private BN statistics "
            "are not accounted by this path."
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    scattering, channels, _ = get_scatter_transform(dataset_name)
    scattering.to(device)

    model = CNNS[dataset_name](
        channels,
        input_norm=input_norm,
        num_groups=num_groups,
        size=None,
    )
    model.to(device)

    train_data_scattered = get_scattered_dataset(
        train_loader,
        scattering,
        device,
        len(train_data),
    )
    test_loader = get_scattered_loader(
        test_loader,
        scattering,
        device,
    )
    optimizer = get_preselected_step_dpsgd_optimizer(
        lr=lr,
        momentum=momentum,
        C_t=C_t,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        p_large=p_large,
        batch_size=batch_size,
        model=model,
        large_step_update=large_step_update,
        large_step_lr_scale=large_step_lr_scale,
        noise_seed=noise_private_seed,
    )
    context = {
        "algorithm": "DP-BiSGD-HF",
        "method": "DP-BiSGD-HF",
        "dataset_name": dataset_name,
        "use_scattering": bool(use_scattering),
        "input_norm": input_norm,
        "num_groups": int(num_groups) if num_groups is not None else None,
        "C_t": float(C_t),
        "lr": float(lr),
        "momentum": float(momentum),
        "large_step_update": large_step_update,
        "large_step_lr_scale": float(large_step_lr_scale),
    }
    if run_context:
        context.update(run_context)

    return train_with_realized_coin_filter(
        train_data=train_data_scattered,
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
        run_context=context,
    )
