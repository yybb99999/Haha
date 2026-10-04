import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.PLD.FindSigmaSmallGMM import compute_delta_projected_gmm_pld
from train_and_validation.train_with_dp import train_with_dp
from train_and_validation.validation import validation


def DPSGD(
    train_data,
    test_data,
    model,
    optimizer,
    batch_size,
    epsilon_budget,
    delta,
    sigma,
    device,
    accountant='rdp',
    target_steps=0,
    sigma_large=15.0,
    p_large=0.05,
    C_t=0.1,
    mixpld_mode='coin_aware',
    skip_pld_check=False,
    prevalidated_delta=None,
    return_metrics=False,
    run_context=None,
):
    minibatch_loader, _ = get_data_loaders_possion(
        minibatch_size=batch_size,
        microbatch_size=1,
        iterations=1,
    )

    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    orders = (
        [1 + x / 10.0 for x in range(1, 100)]
        + list(range(11, 64))
        + [128, 256, 512]
    )

    q = batch_size / len(train_data)

    best_test_acc = 0.0
    best_iter = 0
    epsilon = 0.0
    projected_delta = None
    privacy_status = None

    epsilon_list = []
    test_loss_list = []

    accounted_steps = 0

    test_loss, test_accuracy = validation(model, test_dl, device)

    def run_one_poisson_round():
        train_dl = minibatch_loader(train_data)

        updates_before = optimizer.dp_update_steps

        for _, (data, _) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp(
            model,
            train_dl,
            optimizer,
            device,
        )

        actual_updates = optimizer.dp_update_steps - updates_before

        if actual_updates not in (0, 1):
            raise RuntimeError(
                "Each outer DP-SGD round must perform at most one private update, "
                f"but observed {actual_updates} updates."
            )

        return train_loss, train_accuracy, actual_updates

    if accountant == 'projected_gmm_pld':
        if target_steps <= 0:
            raise ValueError(
                "accountant='projected_gmm_pld' requires target_steps > 0."
            )

        if prevalidated_delta is not None:
            projected_delta = float(prevalidated_delta)
            privacy_status = "reused_find_sigma_small"

        elif skip_pld_check:
            projected_delta = None
            privacy_status = "skipped_by_user"

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
            privacy_status = "validated_now"

        print(
            f"[DPSGD] StepMix privacy status={privacy_status} | "
            f"T={target_steps}, q={q:.6f}, "
            f"sigma_small={sigma:.6f}, sigma_large={sigma_large}, "
            f"p_large={p_large}, eps={epsilon_budget}, "
            f"delta={delta:.3e}, mixpld_mode={mixpld_mode}"
        )

        if projected_delta is not None:
            print(
                f"[DPSGD] projected_delta={projected_delta:.3e}, "
                f"target_delta={delta:.3e}"
            )

            if projected_delta > delta:
                raise ValueError(
                    f"StepMix PLD budget check failed: "
                    f"projected_delta={projected_delta:.3e} > "
                    f"target_delta={delta:.3e}."
                )
        else:
            print(
                "[Privacy] WARNING: PLD validation skipped. "
                "Only use this when the full privacy configuration "
                "has already been verified."
            )

        for accounted_steps in range(1, target_steps + 1):
            train_loss, train_accuracy, actual_updates = run_one_poisson_round()

            test_loss, test_accuracy = validation(model, test_dl, device)

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

    elif accountant == 'rdp':
        while True:
            next_accounted_steps = accounted_steps + 1

            next_epsilon, best_alpha = apply_dp_sgd_analysis(
                q,
                sigma,
                next_accounted_steps,
                orders,
                delta,
            )

            if next_epsilon > epsilon_budget:
                break

            train_loss, train_accuracy, actual_updates = run_one_poisson_round()

            accounted_steps = next_accounted_steps
            epsilon = next_epsilon

            test_loss, test_accuracy = validation(model, test_dl, device)

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

    else:
        raise ValueError(f"Unknown accountant: {accountant}")

    print("------ finished ------")

    print(
        f"[DPSGD] accounted_steps={accounted_steps}, "
        f"actual_dp_updates={optimizer.dp_update_steps}"
    )

    observed_p_large = None

    if hasattr(optimizer, "stepwise_total_steps") and optimizer.stepwise_total_steps > 0:
        observed_p_large = (
            optimizer.stepwise_large_steps / optimizer.stepwise_total_steps
        )

        print(
            f"[Stepwise-GMM] total_steps={optimizer.stepwise_total_steps}, "
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
        "noise_mode": "stepwise_gmm" if accountant == 'projected_gmm_pld' else "gaussian",
        "mixpld_mode": mixpld_mode if accountant == 'projected_gmm_pld' else None,
        "stepwise_total_steps": int(getattr(optimizer, "stepwise_total_steps", 0)),
        "stepwise_large_steps": int(getattr(optimizer, "stepwise_large_steps", 0)),
        "stepwise_small_steps": int(getattr(optimizer, "stepwise_small_steps", 0)),
        "observed_p_large": observed_p_large,
        "projected_delta": projected_delta,
        "privacy_status": privacy_status,
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
