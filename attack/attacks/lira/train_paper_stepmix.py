from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from attacks.lira.train_paper_dpsgd import (
    DebiasedEMA,
    _evaluate,
    _json_safe,
    _load_subset,
    _save_checkpoint,
    _set_seed,
    _write_json,
    lira_learning_rate,
)
from data.util.get_data import get_data
from model.get_model import get_model
from privacy_analysis.PLD.FindSigmaSmallGMM import (
    compute_delta_projected_gmm_pld,
)
from privacy_analysis.RDP.CoinAwareStepMixRDP import (
    epsilon_for_coin_aware_stepmix,
)


class StepMixBranchSampler:
    """Samples one public StepMix branch per private update."""

    def __init__(
        self,
        *,
        sigma_small: float,
        sigma_large: float,
        p_large: float,
        generator: torch.Generator,
    ) -> None:
        if sigma_small <= 0.0:
            raise ValueError("sigma_small must be positive.")
        if sigma_large < sigma_small:
            raise ValueError("sigma_large must be at least sigma_small.")
        if not 0.0 <= p_large <= 1.0:
            raise ValueError("p_large must be in [0, 1].")

        self.sigma_small = float(sigma_small)
        self.sigma_large = float(sigma_large)
        self.p_large = float(p_large)
        self.generator = generator
        self.total_steps = 0
        self.small_steps = 0
        self.large_steps = 0
        self.last_sigma: float | None = None
        self.last_is_large = False

    def sample(self) -> tuple[float, bool]:
        use_large = bool(
            torch.rand((), generator=self.generator, device="cpu").item()
            < self.p_large
        )
        self.total_steps += 1
        self.last_is_large = use_large

        if use_large:
            self.large_steps += 1
            self.last_sigma = self.sigma_large
        else:
            self.small_steps += 1
            self.last_sigma = self.sigma_small

        return self.last_sigma, use_large


def make_stepmix_dp_optimizer_class():
    """Builds the optimizer lazily so importing this module does not require Opacus."""

    try:
        from opacus.optimizers import DPOptimizer
        from opacus.optimizers.optimizer import (
            _check_processed_flag,
            _generate_noise,
            _mark_as_processed,
        )
    except ImportError as exc:
        raise RuntimeError("paper_stepmix requires opacus>=1.6.") from exc

    class StepMixDPOptimizer(DPOptimizer):
        """Opacus per-example clipping with one shared Gaussian branch per step."""

        def __init__(
            self,
            optimizer,
            *,
            sigma_small: float,
            sigma_large: float,
            p_large: float,
            max_grad_norm: float,
            expected_batch_size: int,
            branch_generator: torch.Generator,
            large_step_update: str,
            large_step_lr_scale: float,
            loss_reduction: str = "mean",
            noise_generator=None,
            secure_mode: bool = False,
        ) -> None:
            super().__init__(
                optimizer=optimizer,
                noise_multiplier=sigma_small,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=noise_generator,
                secure_mode=secure_mode,
            )
            if large_step_update not in {
                "normal",
                "sgd_bypass",
                "sgd_bypass_scaled",
                "skip",
            }:
                raise ValueError(f"Unknown large_step_update: {large_step_update}")
            if large_step_lr_scale < 0.0:
                raise ValueError("large_step_lr_scale must be non-negative.")

            self.branch_sampler = StepMixBranchSampler(
                sigma_small=sigma_small,
                sigma_large=sigma_large,
                p_large=p_large,
                generator=branch_generator,
            )
            self.sigma_small = float(sigma_small)
            self.sigma_large = float(sigma_large)
            self.p_large = float(p_large)
            self.large_step_update = large_step_update
            self.large_step_lr_scale = float(large_step_lr_scale)
            self.last_step_sigma: float | None = None
            self.last_step_is_large = False
            self.dp_update_steps = 0

        def add_noise(self) -> None:
            step_sigma, is_large = self.branch_sampler.sample()
            self.last_step_sigma = float(step_sigma)
            self.last_step_is_large = bool(is_large)

            for parameter in self.params:
                _check_processed_flag(parameter.summed_grad)
                noise = _generate_noise(
                    std=step_sigma * self.max_grad_norm,
                    reference=parameter.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                parameter.grad = (parameter.summed_grad + noise).view_as(parameter)
                _mark_as_processed(parameter.summed_grad)

        def _large_step_manual_update(self) -> None:
            if self.large_step_update == "skip":
                return
            if self.large_step_update == "sgd_bypass":
                lr_scale = 1.0
            elif self.large_step_update == "sgd_bypass_scaled":
                lr_scale = self.large_step_lr_scale
            else:
                raise ValueError(
                    "Manual large-step update is valid only for bypass or skip modes."
                )

            with torch.no_grad():
                for group in self.original_optimizer.param_groups:
                    learning_rate = float(group.get("lr", 0.0))
                    for parameter in group["params"]:
                        if parameter.requires_grad and parameter.grad is not None:
                            parameter.add_(
                                parameter.grad,
                                alpha=-learning_rate * lr_scale,
                            )

        def step(
            self,
            closure: Optional[Callable[[], float]] = None,
        ) -> Optional[float]:
            if closure is not None:
                with torch.enable_grad():
                    closure()

            if not self.pre_step():
                return None

            self.dp_update_steps += 1
            if self.last_step_is_large and self.large_step_update != "normal":
                self._large_step_manual_update()
                return None

            return self.original_optimizer.step()

    return StepMixDPOptimizer


def _validate_args(args: argparse.Namespace) -> None:
    if args.algorithm != "DPSGD":
        raise ValueError("paper_stepmix currently supports only algorithm='DPSGD'.")
    if args.dataset_name != "CIFAR-10":
        raise ValueError("paper_stepmix currently supports only CIFAR-10.")
    if args.training_mode != "paper_stepmix":
        raise ValueError("This entrypoint requires training_mode='paper_stepmix'.")
    if args.accountant not in {"projected_gmm_pld", "coin_aware_rdp"}:
        raise ValueError(
            "paper_stepmix requires projected_gmm_pld or coin_aware_rdp."
        )
    if args.noise_mode != "stepwise_gmm":
        raise ValueError("paper_stepmix requires noise_mode='stepwise_gmm'.")
    if args.mixpld_mode != "coin_aware":
        raise ValueError("Formal paper_stepmix runs require coin-aware PLD.")
    if args.stop_rule != "fixed_steps" or args.target_steps <= 0:
        raise ValueError("paper_stepmix requires fixed positive target_steps.")
    if args.audit_train_pool != "train":
        raise ValueError("Paper-aligned CIFAR-10 runs must use audit_train_pool='train'.")
    if args.sampling_scheme != "poisson":
        raise ValueError("paper_stepmix requires Poisson sampling.")
    if args.dp_normalization != "expected_batch":
        raise ValueError("paper_stepmix requires expected-batch normalization.")
    if args.dp_backend != "opacus_stepmix":
        raise ValueError("paper_stepmix requires dp_backend='opacus_stepmix'.")
    if args.weight_decay_scope != "weights_only_post_clip":
        raise ValueError(
            "paper_stepmix requires weight_decay_scope='weights_only_post_clip'."
        )
    if args.poisson_steps_per_epoch != "ceil":
        raise ValueError("paper_stepmix requires poisson_steps_per_epoch='ceil'.")
    if args.sigma_t <= 0.0 or args.C_t <= 0.0:
        raise ValueError("sigma_t and C_t must be positive.")
    if args.sigma_large < args.sigma_t:
        raise ValueError("sigma_large must be at least sigma_t.")
    if not 0.0 <= args.p_large <= 1.0:
        raise ValueError("p_large must be in [0, 1].")
    if args.epsilon is None or args.epsilon <= 0.0:
        raise ValueError("paper_stepmix requires a positive epsilon budget.")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if args.large_step_update not in {
        "normal",
        "sgd_bypass",
        "sgd_bypass_scaled",
        "skip",
    }:
        raise ValueError("Unknown large_step_update.")
    if args.large_step_lr_scale < 0.0:
        raise ValueError("large_step_lr_scale must be non-negative.")
    if args.prevalidated_delta is not None:
        if not math.isfinite(args.prevalidated_delta):
            raise ValueError("prevalidated_delta must be finite.")
        if args.prevalidated_delta > args.delta:
            raise ValueError("prevalidated_delta exceeds the configured delta budget.")
        if args.accountant != "projected_gmm_pld":
            raise ValueError("prevalidated_delta is valid only for projected_gmm_pld.")
    if args.prevalidated_epsilon is not None:
        if not math.isfinite(args.prevalidated_epsilon):
            raise ValueError("prevalidated_epsilon must be finite.")
        if args.prevalidated_epsilon > args.epsilon + 1e-10:
            raise ValueError("prevalidated_epsilon exceeds the epsilon budget.")
        if args.accountant != "coin_aware_rdp":
            raise ValueError("prevalidated_epsilon is valid only for coin_aware_rdp.")


def _privacy_certificate(
    args: argparse.Namespace,
    *,
    sample_rate: float,
) -> dict:
    if args.prevalidated_delta is not None:
        return {
            "projected_delta": float(args.prevalidated_delta),
            "achieved_epsilon": None,
            "optimal_order": None,
            "status": "prevalidated_coin_aware_pld",
        }
    if args.prevalidated_epsilon is not None:
        return {
            "projected_delta": None,
            "achieved_epsilon": float(args.prevalidated_epsilon),
            "optimal_order": args.prevalidated_optimal_order,
            "status": "prevalidated_coin_aware_rdp",
        }
    if args.skip_pld_check:
        return {
            "projected_delta": None,
            "achieved_epsilon": None,
            "optimal_order": None,
            "status": "unchecked_smoke_only",
        }

    if args.accountant == "coin_aware_rdp":
        achieved_epsilon, optimal_order = epsilon_for_coin_aware_stepmix(
            sample_rate=sample_rate,
            steps=args.target_steps,
            delta=args.delta,
            sigma_small=args.sigma_t,
            sigma_large=args.sigma_large,
            p_large=args.p_large,
        )
        if achieved_epsilon > args.epsilon + 1e-10:
            raise RuntimeError(
                f"StepMix RDP budget exceeded: {achieved_epsilon} > {args.epsilon}."
            )
        return {
            "projected_delta": None,
            "achieved_epsilon": float(achieved_epsilon),
            "optimal_order": float(optimal_order),
            "status": "computed_coin_aware_rdp",
        }

    projected_delta = float(
        compute_delta_projected_gmm_pld(
            target_eps=args.epsilon,
            T=args.target_steps,
            q=sample_rate,
            C=args.C_t,
            sigma_small=args.sigma_t,
            sigma_large=args.sigma_large,
            p_large=args.p_large,
            num_z_points=args.pld_num_z_points,
            num_bins=args.pld_num_bins,
            mixpld_mode=args.mixpld_mode,
        )
    )
    if not math.isfinite(projected_delta):
        raise RuntimeError("PLD accountant returned a non-finite projected delta.")
    if projected_delta > args.delta:
        raise RuntimeError(
            f"StepMix privacy budget exceeded: {projected_delta:.12g} > {args.delta:.12g}."
        )
    return {
        "projected_delta": projected_delta,
        "achieved_epsilon": None,
        "optimal_order": None,
        "status": "computed_coin_aware_pld",
    }


def train(args: argparse.Namespace) -> dict:
    _validate_args(args)
    _set_seed(args.seed)

    try:
        import opacus
        from opacus import GradSampleModule
        from opacus.data_loader import DPDataLoader
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise RuntimeError("paper_stepmix requires opacus>=1.6.") from exc

    train_set, test_set, _ = get_data(
        args.dataset_name,
        data_profile=args.data_profile,
        augmentation_mode=args.augmentation_mode,
    )
    train_set, train_indices = _load_subset(
        train_set,
        args.train_indices_path,
        "train",
    )
    test_set, eval_indices = _load_subset(
        test_set,
        args.eval_indices_path,
        "eval",
    )
    if len(train_set) != args.expected_train_size:
        raise ValueError(
            f"Expected {args.expected_train_size} training examples, got {len(train_set)}."
        )

    device = torch.device(args.device)
    model = get_model(
        args.algorithm,
        args.dataset_name,
        device,
        model_arch=args.model_arch,
    )
    validation_errors = ModuleValidator.validate(model, strict=False)
    if validation_errors:
        raise ValueError(f"Model is not Opacus-compatible: {validation_errors}")

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    sample_rate = float(args.batch_size / len(train_set))
    if sample_rate > 1.0:
        raise ValueError("batch_size cannot exceed the paper training-set size.")

    privacy_certificate = _privacy_certificate(
        args,
        sample_rate=sample_rate,
    )
    sampling_generator = torch.Generator()
    sampling_generator.manual_seed(args.seed + 1701)
    branch_generator = torch.Generator()
    branch_generator.manual_seed(args.seed + 2701)
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(args.seed + 3701)
    private_loader = DPDataLoader(
        train_set,
        sample_rate=sample_rate,
        generator=sampling_generator,
        num_workers=0,
        pin_memory=False,
    )
    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    private_loader.batch_sampler.steps = steps_per_epoch
    private_model = GradSampleModule(
        model,
        batch_first=True,
        loss_reduction="mean",
    )
    raw_model = private_model._module
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in raw_model.named_parameters():
        if name.endswith("weight"):
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    base_optimizer = torch.optim.SGD(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.lr,
        momentum=args.momentum,
    )
    optimizer_class = make_stepmix_dp_optimizer_class()
    private_optimizer = optimizer_class(
        optimizer=base_optimizer,
        sigma_small=args.sigma_t,
        sigma_large=args.sigma_large,
        p_large=args.p_large,
        max_grad_norm=args.C_t,
        expected_batch_size=args.batch_size,
        branch_generator=branch_generator,
        large_step_update=args.large_step_update,
        large_step_lr_scale=args.large_step_lr_scale,
        loss_reduction="mean",
        noise_generator=noise_generator,
        secure_mode=args.secure_mode,
    )
    ema = DebiasedEMA(raw_model, args.ema_decay)
    expected_steps = int(args.epochs) * int(steps_per_epoch)
    if expected_steps != args.target_steps:
        raise ValueError(
            "target_steps must equal epochs * len(private_loader): "
            f"expected {expected_steps}, got {args.target_steps}."
        )

    best_acc = float("-inf")
    best_step = 0
    global_step = 0
    last_eval_loss = None
    last_eval_acc = None
    lr_last = None
    sampled_examples = 0
    minimum_batch_size = None
    maximum_batch_size = 0

    for epoch in range(args.epochs):
        private_model.train()
        for batch_index, (data, target) in enumerate(private_loader):
            if args.lr_schedule == "lira_cosine_warmup":
                schedule_position = epoch * len(train_set) + batch_index * args.batch_size
                schedule_total = args.epochs * len(train_set)
                current_lr = lira_learning_rate(args.lr, schedule_position, schedule_total)
            elif args.lr_schedule == "constant":
                current_lr = float(args.lr)
            else:
                raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

            for group in private_optimizer.param_groups:
                group["lr"] = current_lr

            data = data.to(device)
            target = target.to(device)
            private_optimizer.zero_grad(set_to_none=True)
            logits = private_model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            private_optimizer.step()
            ema.update(raw_model)
            current_batch_size = int(target.numel())
            sampled_examples += current_batch_size
            maximum_batch_size = max(maximum_batch_size, current_batch_size)
            minimum_batch_size = (
                current_batch_size
                if minimum_batch_size is None
                else min(minimum_batch_size, current_batch_size)
            )
            global_step += 1
            lr_last = current_lr

        should_evaluate = (
            args.eval_every_epochs > 0
            and ((epoch + 1) % args.eval_every_epochs == 0 or epoch + 1 == args.epochs)
        )
        if should_evaluate:
            raw_state = {
                name: value.detach().clone()
                for name, value in raw_model.state_dict().items()
            }
            raw_model.load_state_dict(ema.state_dict(raw_model))
            last_eval_loss, last_eval_acc = _evaluate(raw_model, test_loader, device)
            raw_model.load_state_dict(raw_state)
            if last_eval_acc > best_acc:
                best_acc = float(last_eval_acc)
                best_step = int(global_step)
            print(
                f"[PaperStepMix] epoch={epoch + 1}/{args.epochs} "
                f"step={global_step}/{args.target_steps} lr={lr_last:.8f} "
                f"ema_test_loss={last_eval_loss:.6f} "
                f"ema_test_acc={last_eval_acc:.4f} "
                f"large_steps={private_optimizer.branch_sampler.large_steps}"
            )

    if global_step != args.target_steps:
        raise RuntimeError(
            f"Training performed {global_step} updates, expected {args.target_steps}."
        )
    if private_optimizer.dp_update_steps != args.target_steps:
        raise RuntimeError("StepMix optimizer/update count mismatch.")
    if private_optimizer.branch_sampler.total_steps != args.target_steps:
        raise RuntimeError("StepMix branch/sample count mismatch.")

    final_ema_state = {
        name: value.detach().cpu()
        for name, value in ema.state_dict(raw_model).items()
    }
    raw_state = {
        name: value.detach().cpu()
        for name, value in raw_model.state_dict().items()
    }
    raw_model.load_state_dict(final_ema_state)
    final_loss, final_acc = _evaluate(raw_model, test_loader, device)
    if final_acc > best_acc:
        best_acc = float(final_acc)
        best_step = int(global_step)

    sampler = private_optimizer.branch_sampler
    metrics = {
        "final_acc": float(final_acc),
        "final_loss": float(final_loss),
        "final_iter": int(global_step),
        "best_acc": float(best_acc),
        "best_iter": int(best_step),
        "epsilon": float(
            privacy_certificate["achieved_epsilon"]
            if privacy_certificate["achieved_epsilon"] is not None
            else args.epsilon
        ),
        "epsilon_budget": float(args.epsilon),
        "privacy_guarantee": args.accountant,
        "privacy_status": privacy_certificate["status"],
        "projected_delta": privacy_certificate["projected_delta"],
        "achieved_epsilon": privacy_certificate["achieved_epsilon"],
        "optimal_order": privacy_certificate["optimal_order"],
        "delta": float(args.delta),
        "accounted_steps": int(global_step),
        "actual_dp_updates": int(private_optimizer.dp_update_steps),
        "actual_updates": int(global_step),
        "training_steps": int(global_step),
        "target_steps": int(args.target_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs": int(args.epochs),
        "train_size": int(len(train_set)),
        "eval_size": int(len(test_set)),
        "sample_rate": sample_rate,
        "batch_size": int(args.batch_size),
        "expected_batch_size": int(private_optimizer.expected_batch_size),
        "mean_sampled_batch_size": float(sampled_examples / global_step),
        "minimum_sampled_batch_size": int(minimum_batch_size),
        "maximum_sampled_batch_size": int(maximum_batch_size),
        "sigma": float(args.sigma_t),
        "sigma_small": float(args.sigma_t),
        "sigma_large": float(args.sigma_large),
        "p_large": float(args.p_large),
        "observed_small_steps": int(sampler.small_steps),
        "observed_large_steps": int(sampler.large_steps),
        "observed_large_fraction": float(sampler.large_steps / sampler.total_steps),
        "large_step_update": args.large_step_update,
        "large_step_lr_scale": float(args.large_step_lr_scale),
        "mixpld_mode": args.mixpld_mode,
        "accountant": args.accountant,
        "C_t": float(args.C_t),
        "lr": float(args.lr),
        "lr_last": float(lr_last),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "ema_decay": float(args.ema_decay),
        "model_arch": args.model_arch,
        "data_profile": args.data_profile,
        "augmentation_mode": args.augmentation_mode,
        "sampling_scheme": args.sampling_scheme,
        "dp_backend": args.dp_backend,
        "dp_normalization": args.dp_normalization,
        "weight_decay_scope": args.weight_decay_scope,
        "poisson_steps_per_epoch": args.poisson_steps_per_epoch,
        "run_tag": args.run_tag,
        "seed": int(args.seed),
        "train_indices_path": args.train_indices_path,
        "eval_indices_path": args.eval_indices_path,
        "train_indices_count": int(len(train_indices)),
        "eval_indices_count": int(len(eval_indices)),
        "opacus_version": opacus.__version__,
        "torch_version": str(torch.__version__),
    }
    _write_json(args.metrics_output_path, metrics)
    _save_checkpoint(
        args.save_model_path,
        model_state_dict=final_ema_state,
        raw_model_state_dict=raw_state,
        metrics=metrics,
        args=args,
    )
    print(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-aligned CIFAR-10 StepMix DP-SGD training."
    )
    parser.add_argument("--training_mode", default="paper_stepmix")
    parser.add_argument("--algorithm", default="DPSGD")
    parser.add_argument("--dataset_name", default="CIFAR-10")
    parser.add_argument("--accountant", default="projected_gmm_pld")
    parser.add_argument("--noise_mode", default="stepwise_gmm")
    parser.add_argument("--mixpld_mode", default="coin_aware")
    parser.add_argument("--sigma_t", type=float, required=True)
    parser.add_argument("--sigma_large", type=float, required=True)
    parser.add_argument("--p_large", type=float, required=True)
    parser.add_argument("--C_t", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--prevalidated_delta", type=float, default=None)
    parser.add_argument("--prevalidated_epsilon", type=float, default=None)
    parser.add_argument("--prevalidated_optimal_order", type=float, default=None)
    parser.add_argument("--skip_pld_check", action="store_true")
    parser.add_argument("--pld_num_z_points", type=int, default=50000)
    parser.add_argument("--pld_num_bins", type=int, default=8192)
    parser.add_argument("--large_step_update", default="sgd_bypass_scaled")
    parser.add_argument("--large_step_lr_scale", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=501)
    parser.add_argument("--target_steps", type=int, required=True)
    parser.add_argument("--stop_rule", default="fixed_steps")
    parser.add_argument("--lr_schedule", default="lira_cosine_warmup")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--model_arch", default="lira_cnn32_3_mean")
    parser.add_argument("--data_profile", default="lira_minus_one_one")
    parser.add_argument("--augmentation_mode", default="weak")
    parser.add_argument("--optimizer_name", default="sgd_momentum")
    parser.add_argument("--sampling_scheme", default="poisson")
    parser.add_argument("--dp_backend", default="opacus_stepmix")
    parser.add_argument("--dp_normalization", default="expected_batch")
    parser.add_argument("--weight_decay_scope", default="weights_only_post_clip")
    parser.add_argument("--poisson_steps_per_epoch", default="ceil")
    parser.add_argument("--audit_train_pool", default="train")
    parser.add_argument("--expected_train_size", type=int, default=25000)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--secure_mode", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--train_indices_path", default=None)
    parser.add_argument("--eval_indices_path", default=None)
    parser.add_argument("--save_model_path", default=None)
    parser.add_argument("--metrics_output_path", default=None)
    return parser


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
