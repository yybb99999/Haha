import os

import torch

from data.util.get_data import (
    get_scatter_transform,
    get_scattered_dataset,
    get_scattered_loader,
)
from data.util.sampling import get_data_loaders_possion
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.PLD.FindSigmaSmallGMM import compute_delta_projected_gmm_pld
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization
from train_and_validation.train_with_dp import train_with_dp
from train_and_validation.validation import validation
from utils.dp_optimizer import DPSGD_Optimizer


CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}


def DPSGD_HF(
    dataset_name,
    train_data,
    test_data,
    model,
    batch_size,
    lr,
    momentum,
    epsilon_budget,
    delta,
    C_t,
    sigma,
    use_scattering,
    input_norm,
    bn_noise_multiplier,
    num_groups,
    device,
    accountant='rdp',
    noise_mode='gaussian',
    target_steps=0,
    sigma_large=15.0,
    p_large=0.05,
    large_step_update='sgd_bypass_scaled',
    large_step_lr_scale=0.1,
    mixpld_mode='coin_aware',
    skip_pld_check=False,
    prevalidated_delta=None,
    return_metrics=False,
    run_context=None,
):
    if accountant not in {'rdp', 'projected_gmm_pld'}:
        raise ValueError(f"Unknown accountant: {accountant}")

    if accountant == 'projected_gmm_pld':
        if noise_mode != 'stepwise_gmm':
            raise ValueError(
                "DPSGD-HF StepMix requires noise_mode='stepwise_gmm'."
            )

        if target_steps <= 0:
            raise ValueError("DPSGD-HF StepMix requires target_steps > 0.")

        if input_norm == 'BN':
            raise ValueError(
                "Formal StepMix PLD cannot be used with input_norm='BN'. "
                "Use GroupNorm or omit input_norm."
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

    if use_scattering:
        scattering, K, _ = get_scatter_transform(dataset_name)
        scattering.to(device)
    else:
        scattering = None
        K = 3 if len(train_data.data.shape) == 4 else 1

    orders = (
        [1 + x / 10.0 for x in range(1, 100)]
        + list(range(11, 64))
        + [128, 256, 512]
    )

    rdp_norm = 0.0

    if input_norm == 'BN':
        save_dir = f"bn_stats/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)

        bn_stats, rdp_norm = scatter_normalization(
            train_loader,
            scattering,
            K,
            device,
            len(train_data),
            len(train_data),
            noise_multiplier=bn_noise_multiplier,
            orders=orders,
            save_dir=save_dir,
        )

        model = CNNS[dataset_name](
            K,
            input_norm='BN',
            bn_stats=bn_stats,
            size=None,
        )
    else:
        model = CNNS[dataset_name](
            K,
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

    minibatch_loader, _ = get_data_loaders_possion(
        minibatch_size=batch_size,
        microbatch_size=1,
        iterations=1,
    )

    optimizer = DPSGD_Optimizer(
        l2_norm_clip=C_t,
        noise_multiplier=sigma,
        minibatch_size=batch_size,
        microbatch_size=1,
        noise_mode=noise_mode,
        sigma_large=sigma_large,
        p_large=p_large,
        large_step_update=large_step_update,
        large_step_lr_scale=large_step_lr_scale,
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
    )

    q = batch_size / len(train_data_scattered)

    best_test_acc = 0.0
    best_iter = 0
    accounted_steps = 0
    epsilon = 0.0
    projected_delta = None
    privacy_status = None

    epsilon_list = []
    test_loss_list = []

    test_loss, test_accuracy = validation(model, test_loader, device)

    def run_one_poisson_round():
        train_dl = minibatch_loader(train_data_scattered)

        updates_before = optimizer.dp_update_steps

        for _, (data, _) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp(
            model,
            train_dl,
            optimizer,
            device,
        )

        completed_updates = optimizer.dp_update_steps - updates_before

        if completed_updates not in (0, 1):
            raise RuntimeError(
                "Each DPSGD-HF outer round must execute at most one "
                f"private update, but observed {completed_updates}."
            )

        return train_loss, train_accuracy, completed_updates

    if accountant == 'projected_gmm_pld':
        if prevalidated_delta is not None:
            projected_delta = float(prevalidated_delta)
            privacy_status = 'reused_find_sigma_small'

        elif skip_pld_check:
            projected_delta = None
            privacy_status = 'skipped_by_user'

        else:
            projected_delta = compute_delta_projected_gmm_pld(
                target_eps=epsilon_budget,
                T=target_steps,
                q=q,
                C=C_t,
                sigma_small=sigma,
                sigma_large=sigma_large,
                p_large=p_large,
                mixpld_mode=mixpld_mode,
            )
            privacy_status = 'validated_now'

        print(
            f"[DPSGD-HF] StepMix privacy status={privacy_status} | "
            f"T={target_steps}, q={q:.6f}, "
            f"sigma_small={sigma:.6f}, "
            f"sigma_large={sigma_large}, "
            f"p_large={p_large}, "
            f"eps={epsilon_budget}, "
            f"delta={delta:.3e}, "
            f"mixpld_mode={mixpld_mode}, "
            f"input_norm={input_norm}"
        )

        if projected_delta is not None:
            print(
                f"[DPSGD-HF] projected_delta={projected_delta:.3e}, "
                f"target_delta={delta:.3e}"
            )

            if projected_delta > delta:
                raise ValueError(
                    "DPSGD-HF StepMix PLD budget check failed: "
                    f"projected_delta={projected_delta:.3e} > "
                    f"target_delta={delta:.3e}."
                )
        else:
            print(
                "[DPSGD-HF] WARNING: PLD validation skipped. "
                "Use this only for an already validated privacy tuple."
            )

        for accounted_steps in range(1, target_steps + 1):
            run_one_poisson_round()

            test_loss, test_accuracy = validation(
                model,
                test_loader,
                device,
            )

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_iter = accounted_steps

            epsilon_list.append(torch.tensor(float(epsilon_budget)))
            test_loss_list.append(test_loss)

            print(
                f"iters:{accounted_steps}/{target_steps}, "
                f"accountant:stepwise_mixpld, "
                f"epsilon:{epsilon_budget:.4f}, "
                f"delta:{delta:.1e}, "
                f"actual_updates:{optimizer.dp_update_steps} | "
                f"Test set: Average loss:{test_loss:.4f}, "
                f"Accuracy:({test_accuracy:.2f}%)"
            )

    else:
        while True:
            next_steps = accounted_steps + 1

            rdp_train = compute_rdp(
                q,
                sigma,
                next_steps,
                orders,
            )

            next_epsilon, _ = compute_eps(
                orders,
                rdp_train + rdp_norm,
                delta,
            )

            if next_epsilon > epsilon_budget:
                break

            run_one_poisson_round()

            accounted_steps = next_steps
            epsilon = next_epsilon

            test_loss, test_accuracy = validation(
                model,
                test_loader,
                device,
            )

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_iter = accounted_steps

            epsilon_list.append(torch.tensor(float(epsilon)))
            test_loss_list.append(test_loss)

            print(
                f"iters:{accounted_steps}, "
                f"accountant:rdp, "
                f"epsilon:{epsilon:.4f}, "
                f"delta:{delta:.1e}, "
                f"actual_updates:{optimizer.dp_update_steps} | "
                f"Test set: Average loss:{test_loss:.4f}, "
                f"Accuracy:({test_accuracy:.2f}%)"
            )

    print("------ finished ------")

    print(
        f"[DPSGD-HF] accounted_steps={accounted_steps}, "
        f"actual_dp_updates={optimizer.dp_update_steps}"
    )

    observed_p_large = None

    if optimizer.stepwise_total_steps > 0:
        observed_p_large = (
            optimizer.stepwise_large_steps
            / optimizer.stepwise_total_steps
        )

        print(
            f"[Stepwise-GMM] "
            f"total_steps={optimizer.stepwise_total_steps}, "
            f"small_steps={optimizer.stepwise_small_steps}, "
            f"large_steps={optimizer.stepwise_large_steps}, "
            f"observed_p_large={observed_p_large:.6f}"
        )

    metrics = {
        "final_acc": float(test_accuracy),
        "final_iter": int(accounted_steps),
        "best_acc": float(best_test_acc),
        "best_iter": int(best_iter),
        "epsilon": float(epsilon_budget if accountant == 'projected_gmm_pld' else epsilon),
        "delta": float(delta),
        "accountant": accountant,
        "accounted_steps": int(accounted_steps),
        "actual_dp_updates": int(optimizer.dp_update_steps),
        "sampling": "poisson",
        "sample_rate": float(q),
        "batch_size": int(batch_size),
        "microbatch_size": 1,
        "target_steps": int(target_steps),
        "C_t": float(C_t),
        "sigma": float(sigma) if accountant == 'rdp' else None,
        "sigma_small": float(sigma) if accountant == 'projected_gmm_pld' else None,
        "sigma_large": float(sigma_large) if accountant == 'projected_gmm_pld' else None,
        "p_large": float(p_large) if accountant == 'projected_gmm_pld' else None,
        "noise_mode": noise_mode,
        "mixpld_mode": mixpld_mode if accountant == 'projected_gmm_pld' else None,
        "large_step_update": large_step_update if accountant == 'projected_gmm_pld' else None,
        "large_step_lr_scale": float(large_step_lr_scale) if accountant == 'projected_gmm_pld' else None,
        "stepwise_total_steps": int(getattr(optimizer, "stepwise_total_steps", 0)),
        "stepwise_large_steps": int(getattr(optimizer, "stepwise_large_steps", 0)),
        "stepwise_small_steps": int(getattr(optimizer, "stepwise_small_steps", 0)),
        "observed_p_large": observed_p_large,
        "projected_delta": projected_delta,
        "privacy_status": privacy_status,
        "input_norm": input_norm,
        "use_scattering": bool(use_scattering),
        "num_groups": int(num_groups) if num_groups is not None else None,
        "bn_noise_multiplier": float(bn_noise_multiplier) if bn_noise_multiplier is not None else None,
    }

    if run_context:
        metrics.update(run_context)

    if return_metrics:
        return (
            test_accuracy,
            accounted_steps,
            best_test_acc,
            best_iter,
            model,
            [epsilon_list, test_loss_list],
            metrics,
        )

    return (
        test_accuracy,
        accounted_steps,
        best_test_acc,
        best_iter,
        model,
        [epsilon_list, test_loss_list],
    )
