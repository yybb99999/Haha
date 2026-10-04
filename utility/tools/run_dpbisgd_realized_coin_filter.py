"""Standalone runner for privacy-filtered DP-BiSGD and DP-BiSGD-HF."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from algorithm.DPSGD_HF_RealizedCoinFilter import (  
    DPSGD_HF_RealizedCoinFilter,
)
from algorithm.DPSGD_RealizedCoinFilter import (  
    DPSGD_RealizedCoinFilter,
)
from data.util.get_data import get_data  
from model.get_model import get_model  
from utils.preselected_step_dp_optimizer import (  
    get_preselected_step_dpsgd_optimizer,
)
from utils.realized_filter_integrity import verify_legacy_core  


ALGORITHM_ALIASES = {
    "DP-BiSGD": "DP-BiSGD",
    "DP-BiSGD-HF": "DP-BiSGD-HF",
    "DPSGD": "DP-BiSGD",
    "DPSGD-HF": "DP-BiSGD-HF",
}


SOURCE_FILES = (
    "algorithm/DPSGD.py",
    "algorithm/DPSGD_HF.py",
    "utils/dp_optimizer.py",
    "privacy_analysis/PLD/FindSigmaSmallGMM.py",
    "algorithm/realized_coin_filter_training.py",
    "algorithm/DPSGD_RealizedCoinFilter.py",
    "algorithm/DPSGD_HF_RealizedCoinFilter.py",
    "utils/preselected_step_dp_optimizer.py",
    "utils/realized_coin_sampling.py",
    "utils/realized_filter_integrity.py",
    "privacy_analysis/filters/realized_coin_rdp_filter.py",
    "privacy_analysis/filters/pessimistic_coin_aware_pld_filter.py",
    "tools/run_dpbisgd_realized_coin_filter.py",
    "tools/validate_dpbisgd_realized_coin_run.py",
    "requirements-realized-filter.txt",
)


def _set_public_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_hashes():
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        result[relative] = _sha256(path) if path.is_file() else None
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_result_csv(path, metrics):
    fields = [
        "Best_Acc",
        "Fin_Acc",
        "Sigma_Small",
        "Sigma_Large",
        "P_Large",
        "Best_Acc_t",
        "Fin_Acc_T",
        "status",
        "stop_reason",
        "algorithm",
        "dataset_name",
        "seed",
        "epsilon_target",
        "delta_target",
        "p_large",
        "sigma_small",
        "sigma_large",
        "T_final",
        "stepwise_small_steps",
        "stepwise_large_steps",
        "observed_p_large",
        "epsilon_upper_final",
        "delta_upper_final_at_target_epsilon",
        "best_acc",
        "best_iter",
        "final_acc",
        "final_iter",
        "batch_size",
        "sample_rate",
        "C_t",
        "lr",
        "momentum",
        "large_step_update",
        "large_step_lr_scale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics.get(field) for field in fields})


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train DP-BiSGD with a realized public-coin pessimistic PLD filter."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHM_ALIASES),
        required=True,
        help=(
            "Public method name. DPSGD and DPSGD-HF are retained only as "
            "backward-compatible aliases."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        choices=["MNIST", "FMNIST", "CIFAR-10"],
        required=True,
    )
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--sigma_small", type=float, required=True)
    parser.add_argument("--sigma_large", type=float, required=True)
    parser.add_argument("--p_large", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--C_t", type=float, default=0.1)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--large_step_update",
        choices=["normal", "sgd_bypass", "sgd_bypass_scaled", "skip"],
        default="sgd_bypass_scaled",
    )
    parser.add_argument("--large_step_lr_scale", type=float, default=0.1)
    parser.add_argument("--max_updates_safety", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch_seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--use_scattering", action="store_true")
    parser.add_argument(
        "--input_norm",
        choices=["GroupNorm", "BN"],
        default=None,
    )
    parser.add_argument("--num_groups", type=int, default=27)
    parser.add_argument("--pld_discretization", type=float, default=1e-4)
    parser.add_argument("--pld_log_mass_truncation", type=float, default=-50.0)
    parser.add_argument("--pld_tail_mass_truncation", type=float, default=1e-15)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--save_model", action="store_true")
    return parser


def _run(args, output_dir):
    verified_legacy_hashes = verify_legacy_core(ROOT)
    cli_algorithm = args.algorithm
    method = ALGORITHM_ALIASES[cli_algorithm]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if method == "DP-BiSGD-HF":
        if not args.use_scattering or args.input_norm != "GroupNorm":
            raise ValueError(
                "DP-BiSGD-HF requires --use_scattering --input_norm GroupNorm."
            )
    elif args.use_scattering or args.input_norm is not None:
        raise ValueError(
            "Scattering and input_norm options are reserved for DP-BiSGD-HF."
        )
    if args.sigma_large < args.sigma_small:
        raise ValueError("sigma_large must be at least sigma_small.")

    _set_public_seed(args.seed)
    branch_seed = args.branch_seed
    if branch_seed is None:
        branch_seed = args.seed + 1_000_003

    command_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": {**vars(args), "algorithm": method},
        "algorithm_cli": cli_algorithm,
        "method": method,
        "resolved_branch_seed": branch_seed,
        "private_rng_policy": (
            "Poisson sampling and Gaussian noise use independent OS-seeded "
            "generators. Only SHA256 seed commitments are reported."
        ),
        "source_sha256": _source_hashes(),
        "verified_legacy_sha256": verified_legacy_hashes,
    }
    _atomic_json(output_dir / "command.json", command_payload)

    train_data, test_data, _ = get_data(args.dataset_name, augment=False)
    run_context = {
        "algorithm": method,
        "algorithm_cli": cli_algorithm,
        "method": method,
        "dataset_name": args.dataset_name,
        "seed": int(args.seed),
        "run_tag": args.run_tag,
        "C_t": float(args.C_t),
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "large_step_update": args.large_step_update,
        "large_step_lr_scale": float(args.large_step_lr_scale),
        "device": args.device,
        "train_size": len(train_data),
        "eval_size": len(test_data),
    }

    trace_path = output_dir / "privacy_trace.jsonl"
    with trace_path.open("x", encoding="utf-8", buffering=1) as trace_handle:
        def trace_callback(event):
            trace_handle.write(json.dumps(_json_safe(event), sort_keys=True) + "\n")
            trace_handle.flush()

        noise_private_seed = secrets.randbits(63)
        poisson_private_seed = secrets.randbits(63)
        if method == "DP-BiSGD":
            model = get_model("DPSGD", args.dataset_name, args.device)
            optimizer = get_preselected_step_dpsgd_optimizer(
                lr=args.lr,
                momentum=args.momentum,
                C_t=args.C_t,
                sigma_small=args.sigma_small,
                sigma_large=args.sigma_large,
                p_large=args.p_large,
                batch_size=args.batch_size,
                model=model,
                large_step_update=args.large_step_update,
                large_step_lr_scale=args.large_step_lr_scale,
                noise_seed=noise_private_seed,
            )
            result = DPSGD_RealizedCoinFilter(
                train_data=train_data,
                test_data=test_data,
                model=model,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epsilon_budget=args.epsilon,
                delta=args.delta,
                sigma_small=args.sigma_small,
                sigma_large=args.sigma_large,
                p_large=args.p_large,
                device=args.device,
                max_updates_safety=args.max_updates_safety,
                branch_seed=branch_seed,
                poisson_private_seed=poisson_private_seed,
                pld_discretization=args.pld_discretization,
                pld_log_mass_truncation=args.pld_log_mass_truncation,
                pld_tail_mass_truncation=args.pld_tail_mass_truncation,
                trace_callback=trace_callback,
                run_context=run_context,
            )
        else:
            result = DPSGD_HF_RealizedCoinFilter(
                dataset_name=args.dataset_name,
                train_data=train_data,
                test_data=test_data,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                epsilon_budget=args.epsilon,
                delta=args.delta,
                C_t=args.C_t,
                sigma_small=args.sigma_small,
                sigma_large=args.sigma_large,
                p_large=args.p_large,
                use_scattering=args.use_scattering,
                input_norm=args.input_norm,
                num_groups=args.num_groups,
                device=args.device,
                max_updates_safety=args.max_updates_safety,
                branch_seed=branch_seed,
                poisson_private_seed=poisson_private_seed,
                noise_private_seed=noise_private_seed,
                large_step_update=args.large_step_update,
                large_step_lr_scale=args.large_step_lr_scale,
                pld_discretization=args.pld_discretization,
                pld_log_mass_truncation=args.pld_log_mass_truncation,
                pld_tail_mass_truncation=args.pld_tail_mass_truncation,
                trace_callback=trace_callback,
                run_context=run_context,
            )

    final_acc, final_iter, best_acc, best_iter, model, iter_list, metrics = result
    metrics.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "source_sha256": _source_hashes(),
        }
    )
    _atomic_json(output_dir / "metrics.json", metrics)
    _write_result_csv(output_dir / "result.csv", metrics)
    torch.save(iter_list, output_dir / "iterList.pth")
    if args.save_model:
        torch.save(
            {"state_dict": model.state_dict(), "metrics": metrics},
            output_dir / "model.pth",
        )

    artifact_names = [
        "command.json",
        "metrics.json",
        "result.csv",
        "privacy_trace.jsonl",
        "iterList.pth",
    ]
    if args.save_model:
        artifact_names.append("model.pth")
    manifest = {
        "status": metrics["status"],
        "artifacts": {
            name: _sha256(output_dir / name) for name in artifact_names
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"Finished status={metrics['status']} stop={metrics['stop_reason']} "
        f"T_final={final_iter} best={best_acc:.2f}@{best_iter} "
        f"final={final_acc:.2f}"
    )
    return metrics["status"]


def main():
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        status = _run(args, output_dir)
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output_dir / "failure.json", failure)
        raise
    if status != "passed":
        print(
            "The safety cap was reached before the privacy filter stopped. "
            "The artifacts are marked incomplete and must not be reported."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
