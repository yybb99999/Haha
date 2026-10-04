from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


def _extract_logits(output):
    if isinstance(output, tuple):
        return output[0]

    if hasattr(output, "logits"):
        return output.logits

    return output


@torch.no_grad()
def collect_logits(model, data_loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    logits_parts = []
    label_parts = []

    for data, labels in data_loader:
        data = data.to(device)
        output = _extract_logits(model(data))

        logits_parts.append(output.detach().cpu())
        label_parts.append(labels.detach().cpu())

    if not logits_parts:
        raise ValueError("Cannot collect logits from an empty data loader.")

    logits = torch.cat(logits_parts, dim=0).numpy()
    labels = torch.cat(label_parts, dim=0).numpy()

    if not np.all(np.isfinite(logits)):
        raise ValueError("Collected logits contain NaN or Infinity.")

    return logits, labels


@torch.no_grad()
def collect_augmented_logits(
    model,
    data_loader,
    device,
    query_augmentations,
) -> tuple[np.ndarray, np.ndarray]:
    augmentations = tuple(str(item) for item in query_augmentations)

    if not augmentations:
        raise ValueError("query_augmentations cannot be empty.")

    invalid = set(augmentations) - {"identity", "hflip"}

    if invalid:
        raise ValueError(f"Unsupported query augmentations: {sorted(invalid)}")

    model.eval()
    logits_parts = []
    label_parts = []

    for data, labels in data_loader:
        data = data.to(device)
        outputs = []

        for augmentation in augmentations:
            query = data if augmentation == "identity" else torch.flip(data, dims=(-1,))
            outputs.append(_extract_logits(model(query)).detach().cpu())

        logits_parts.append(torch.stack(outputs, dim=1))
        label_parts.append(labels.detach().cpu())

    if not logits_parts:
        raise ValueError("Cannot collect logits from an empty data loader.")

    logits = torch.cat(logits_parts, dim=0).numpy()
    labels = torch.cat(label_parts, dim=0).numpy()

    if not np.all(np.isfinite(logits)):
        raise ValueError("Collected logits contain NaN or Infinity.")

    return logits, labels


def logit_scaled_confidence(logits, labels) -> np.ndarray:
    """Compute log(p_y / (1 - p_y)) directly from raw logits."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long).reshape(-1)

    if logits_tensor.ndim not in (2, 3):
        raise ValueError(
            "logits must have shape [N, C] or [N, A, C], "
            f"got {tuple(logits_tensor.shape)}"
        )

    if logits_tensor.ndim == 2:
        num_samples, num_classes = logits_tensor.shape

        if num_classes < 2:
            raise ValueError("LiRA requires at least two output classes.")

        if labels_tensor.shape[0] != num_samples:
            raise ValueError("logits and labels must have the same number of samples.")

        if torch.any(labels_tensor < 0) or torch.any(labels_tensor >= num_classes):
            raise ValueError("labels contain an invalid class index.")

        if not torch.isfinite(logits_tensor).all():
            raise ValueError("logits contain NaN or Infinity.")

        row_indices = torch.arange(num_samples)
        true_logits = logits_tensor[row_indices, labels_tensor]
        wrong_mask = torch.ones_like(logits_tensor, dtype=torch.bool)
        wrong_mask[row_indices, labels_tensor] = False
        wrong_logits = logits_tensor.masked_fill(~wrong_mask, float("-inf"))
        scores = true_logits - torch.logsumexp(wrong_logits, dim=1)
    else:
        num_samples, num_augmentations, num_classes = logits_tensor.shape

        if num_augmentations <= 0:
            raise ValueError("LiRA requires at least one query augmentation.")

        if num_classes < 2:
            raise ValueError("LiRA requires at least two output classes.")

        if labels_tensor.shape[0] != num_samples:
            raise ValueError("logits and labels must have the same number of samples.")

        if torch.any(labels_tensor < 0) or torch.any(labels_tensor >= num_classes):
            raise ValueError("labels contain an invalid class index.")

        if not torch.isfinite(logits_tensor).all():
            raise ValueError("logits contain NaN or Infinity.")

        label_index = labels_tensor[:, None, None].expand(
            num_samples, num_augmentations, 1
        )
        true_logits = torch.gather(logits_tensor, 2, label_index).squeeze(2)
        log_normalizer = torch.logsumexp(logits_tensor, dim=2)
        wrong_logits = logits_tensor.clone()
        wrong_logits.scatter_(2, label_index, float("-inf"))
        wrong_logsumexp = torch.logsumexp(wrong_logits, dim=2)
        log_query_count = float(np.log(num_augmentations))
        log_mean_true_probability = (
            torch.logsumexp(true_logits - log_normalizer, dim=1)
            - log_query_count
        )
        log_mean_wrong_probability = (
            torch.logsumexp(wrong_logsumexp - log_normalizer, dim=1)
            - log_query_count
        )
        scores = log_mean_true_probability - log_mean_wrong_probability

    if not torch.isfinite(scores).all():
        raise ValueError("logit-scaled confidence contains non-finite values.")

    return scores.cpu().numpy().astype(np.float64)


def save_logits(path: Union[str, Path], logits, labels, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim not in (2, 3) or labels.reshape(-1).shape[0] != logits.shape[0]:
        raise ValueError("Invalid logits or labels shape.")

    if not np.all(np.isfinite(logits)):
        raise ValueError("Refusing to save non-finite logits.")

    payload = {
        "logits": logits,
        "labels": labels,
    }

    if extra:
        payload.update(extra)

    np.savez_compressed(path, **payload)


def load_logits(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
