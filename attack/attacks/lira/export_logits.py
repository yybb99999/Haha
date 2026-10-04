from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attacks.common.config import sha256_file
from attacks.common.logits import collect_augmented_logits, collect_logits, save_logits
from data.util.get_data import (
    get_data,
    get_scatter_transform,
    get_scattered_loader,
)
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from model.get_model import get_model


HF_CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}


def _load_indices(path):
    indices = np.asarray(np.load(path), dtype=np.int64).reshape(-1)

    if indices.size == 0:
        raise ValueError("Query indices cannot be empty.")

    if np.unique(indices).size != indices.size:
        raise ValueError("Query indices contain duplicates.")

    return indices


def _load_checkpoint(path, device):
    payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("args", {}), payload.get("metrics", {})

    return payload, {}, {}


def _build_model(args_dict, cli_args, device):
    algorithm = args_dict.get("algorithm", cli_args.algorithm)
    dataset_name = args_dict.get("dataset_name", cli_args.dataset_name)

    if algorithm == "DPSGD-HF":
        if args_dict.get("input_norm") == "BN":
            raise ValueError("BN checkpoint export is not supported without saved BN stats.")

        use_scattering = bool(args_dict.get("use_scattering", cli_args.use_scattering))

        if use_scattering:
            _, k_channels, _ = get_scatter_transform(dataset_name)
        else:
            k_channels = 3 if dataset_name == "CIFAR-10" else 1

        model = HF_CNNS[dataset_name](
            k_channels,
            input_norm=args_dict.get("input_norm", cli_args.input_norm),
            num_groups=int(args_dict.get("num_groups", cli_args.num_groups)),
            size=None,
        )
    else:
        model = get_model(
            algorithm,
            dataset_name,
            device,
            model_arch=args_dict.get("model_arch"),
        )

    model.load_state_dict(_state_dict_without_module_prefix(cli_args, model, args_dict))
    model.to(device)
    model.eval()
    return model


def _state_dict_without_module_prefix(cli_args, model, args_dict):
    state_dict, _, _ = _load_checkpoint(cli_args.checkpoint_path, cli_args.device)
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    return cleaned


def export_checkpoint_logits(
    *,
    checkpoint_path: str,
    output_path: str,
    dataset_name: str,
    algorithm: str,
    indices_path: str,
    batch_size: int,
    device: str,
    use_scattering: bool = False,
    input_norm: Optional[str] = None,
    num_groups: int = 27,
    candidate_pool: str = "train",
    query_augmentations: Optional[list[str]] = None,
) -> None:
    query_augmentations = list(query_augmentations or ["identity"])
    cli_args = argparse.Namespace(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        dataset_name=dataset_name,
        algorithm=algorithm,
        indices_path=indices_path,
        batch_size=batch_size,
        device=device,
        use_scattering=use_scattering,
        input_norm=input_norm,
        num_groups=num_groups,
        candidate_pool=candidate_pool,
        query_augmentations=query_augmentations,
    )

    state_dict, checkpoint_args, checkpoint_metrics = _load_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint_args or {}
    checkpoint_args.setdefault("algorithm", algorithm)
    checkpoint_args.setdefault("dataset_name", dataset_name)

    model = _build_model(checkpoint_args, cli_args, device)
    train_data, test_data, combined_data = get_data(
        dataset_name,
        augment=False,
        data_profile=checkpoint_args.get(
            "data_profile",
            "legacy_imagenet_norm",
        ),
        augmentation_mode="none",
    )

    if candidate_pool == "train":
        candidate_data = train_data
    elif candidate_pool == "test":
        candidate_data = test_data
    elif candidate_pool == "combined":
        candidate_data = combined_data
    else:
        raise ValueError("candidate_pool must be one of train, test, combined.")

    indices = _load_indices(indices_path)

    if np.any(indices < 0) or np.any(indices >= len(candidate_data)):
        raise ValueError("Query indices are outside the selected candidate pool.")

    subset = torch.utils.data.Subset(candidate_data, indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    if checkpoint_args.get("algorithm", algorithm) == "DPSGD-HF":
        if bool(checkpoint_args.get("use_scattering", use_scattering)):
            if query_augmentations != ["identity"]:
                raise ValueError(
                    "Augmented queries are not supported for scattering checkpoints."
                )
            scattering, _, _ = get_scatter_transform(dataset_name)
            scattering.to(device)
        else:
            scattering = None

        loader = get_scattered_loader(loader, scattering, device)

    if query_augmentations == ["identity"]:
        logits, labels = collect_logits(model, loader, device)
    else:
        logits, labels = collect_augmented_logits(
            model,
            loader,
            device,
            query_augmentations,
        )
    save_logits(
        output_path,
        logits,
        labels,
        extra={
            "query_indices": indices,
            "checkpoint_path": np.asarray(str(checkpoint_path)),
            "checkpoint_sha256": np.asarray(sha256_file(checkpoint_path)),
            "indices_path": np.asarray(str(indices_path)),
            "indices_sha256": np.asarray(sha256_file(indices_path)),
            "candidate_pool": np.asarray(str(candidate_pool)),
            "query_augmentations": np.asarray(query_augmentations),
        },
    )
    print(f"[OfflineLiRA] Saved logits to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export logits from a saved audit checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--indices_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_scattering", action="store_true")
    parser.add_argument("--input_norm", type=str, default=None)
    parser.add_argument("--num_groups", type=int, default=27)
    parser.add_argument("--candidate_pool", type=str, default="train", choices=["train", "test", "combined"])
    parser.add_argument(
        "--query_augmentations",
        type=str,
        nargs="+",
        default=["identity"],
        choices=["identity", "hflip"],
    )
    args = parser.parse_args()

    export_checkpoint_logits(**vars(args))


if __name__ == "__main__":
    main()
