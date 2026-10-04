from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.util.get_data import get_data
from model.get_model import get_model


def lira_learning_rate(base_lr: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    warmup = min(progress * 100.0, 1.0)
    return float(base_lr) * math.cos(progress * 7.0 * math.pi / 16.0) * warmup


class DebiasedEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")

        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach(),
                alpha=1.0 - self.decay,
            )

        self.num_updates += 1

    def state_dict(self, model: torch.nn.Module) -> dict:
        state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

        if self.num_updates == 0:
            return state

        correction = 1.0 - self.decay ** self.num_updates

        for name, value in self.shadow.items():
            state[name] = (value / correction).detach().clone()

        return state


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_subset(dataset, indices_path: str | None, label: str):
    if indices_path is None:
        return dataset, np.arange(len(dataset), dtype=np.int64)

    indices = np.asarray(np.load(indices_path), dtype=np.int64).reshape(-1)

    if indices.size == 0:
        raise ValueError(f"{label} indices cannot be empty.")

    if np.unique(indices).size != indices.size:
        raise ValueError(f"{label} indices contain duplicates.")

    if np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise ValueError(f"{label} indices are outside the selected dataset.")

    return torch.utils.data.Subset(dataset, indices.tolist()), indices


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        total_loss += float(F.cross_entropy(logits, target, reduction="sum").item())
        total_correct += int((logits.argmax(dim=1) == target).sum().item())
        total_examples += int(target.numel())

    if total_examples == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    return total_loss / total_examples, 100.0 * total_correct / total_examples


def _json_safe(value):
    if isinstance(value, str):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _write_json(path: str | None, payload: dict) -> None:
    if path is None:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_checkpoint(
    path: str | None,
    *,
    model_state_dict: dict,
    raw_model_state_dict: dict,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    if path is None:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "raw_model_state_dict": raw_model_state_dict,
            "metrics": _json_safe(metrics),
            "args": _json_safe(vars(args)),
        },
        output,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.algorithm != "DPSGD":
        raise ValueError("paper_dp currently supports only algorithm='DPSGD'.")

    if args.dataset_name != "CIFAR-10":
        raise ValueError("paper_dp currently supports only CIFAR-10.")

    if args.training_mode != "paper_dp":
        raise ValueError("This entrypoint requires training_mode='paper_dp'.")

    if args.accountant != "rdp" or args.noise_mode != "gaussian":
        raise ValueError("paper_dp baseline requires RDP accounting and Gaussian noise.")

    if args.stop_rule != "fixed_steps" or args.target_steps <= 0:
        raise ValueError("paper_dp requires stop_rule='fixed_steps' and target_steps > 0.")

    if args.audit_train_pool != "train":
        raise ValueError("Paper-aligned CIFAR-10 runs must use audit_train_pool='train'.")

    if args.sampling_scheme != "poisson":
        raise ValueError("paper_dp requires Poisson sampling.")

    if args.dp_normalization != "expected_batch":
        raise ValueError("paper_dp requires expected-batch gradient normalization.")

    if args.dp_backend != "opacus":
        raise ValueError("paper_dp baseline requires dp_backend='opacus'.")

    if args.weight_decay_scope != "weights_only_post_clip":
        raise ValueError(
            "paper_dp requires weight_decay_scope='weights_only_post_clip'."
        )

    if args.poisson_steps_per_epoch != "ceil":
        raise ValueError("paper_dp requires poisson_steps_per_epoch='ceil'.")

    if args.sigma_t < 0.0 or args.C_t <= 0.0:
        raise ValueError("sigma_t must be non-negative and C_t must be positive.")


def train(args: argparse.Namespace) -> dict:
    _validate_args(args)
    _set_seed(args.seed)

    try:
        from opacus import GradSampleModule
        from opacus.accountants import RDPAccountant
        from opacus.data_loader import DPDataLoader
        from opacus.optimizers import DPOptimizer
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise RuntimeError("paper_dp requires opacus>=1.6.") from exc

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

    sampling_generator = torch.Generator()
    sampling_generator.manual_seed(args.seed + 1701)
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
            {
                "params": decay_parameters,
                "weight_decay": args.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        momentum=args.momentum,
    )
    private_optimizer = DPOptimizer(
        optimizer=base_optimizer,
        noise_multiplier=args.sigma_t,
        max_grad_norm=args.C_t,
        expected_batch_size=args.batch_size,
        loss_reduction="mean",
        secure_mode=args.secure_mode,
    )
    accountant = RDPAccountant()
    private_optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
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
                current_lr = lira_learning_rate(
                    args.lr,
                    schedule_position,
                    schedule_total,
                )
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
            and (
                (epoch + 1) % args.eval_every_epochs == 0
                or epoch + 1 == args.epochs
            )
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
                f"[PaperDP] epoch={epoch + 1}/{args.epochs} "
                f"step={global_step}/{args.target_steps} lr={lr_last:.8f} "
                f"ema_test_loss={last_eval_loss:.6f} "
                f"ema_test_acc={last_eval_acc:.4f}"
            )

    if global_step != args.target_steps:
        raise RuntimeError(
            f"Training performed {global_step} updates, expected {args.target_steps}."
        )

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

    if args.sigma_t == 0.0:
        privacy_epsilon = None
        privacy_label = "infinity_sigma_zero"
    else:
        try:
            privacy_epsilon = float(accountant.get_epsilon(delta=args.delta))
        except Exception as exc:
            privacy_epsilon = None
            privacy_label = f"rdp_unavailable:{type(exc).__name__}"
        else:
            privacy_label = "rdp"

    metrics = {
        "final_acc": float(final_acc),
        "final_loss": float(final_loss),
        "final_iter": int(global_step),
        "best_acc": float(best_acc),
        "best_iter": int(best_step),
        "epsilon": privacy_epsilon,
        "epsilon_budget": args.epsilon,
        "privacy_guarantee": privacy_label,
        "delta": float(args.delta),
        "accounted_steps": int(global_step),
        "actual_dp_updates": int(global_step),
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
        "opacus_version": __import__("opacus").__version__,
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
        description="Paper-aligned CIFAR-10 Gaussian DP-SGD training."
    )
    parser.add_argument("--training_mode", default="paper_dp")
    parser.add_argument("--algorithm", default="DPSGD")
    parser.add_argument("--dataset_name", default="CIFAR-10")
    parser.add_argument("--accountant", default="rdp")
    parser.add_argument("--noise_mode", default="gaussian")
    parser.add_argument("--sigma_t", type=float, required=True)
    parser.add_argument("--C_t", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-5)
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
    parser.add_argument("--dp_backend", default="opacus")
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
