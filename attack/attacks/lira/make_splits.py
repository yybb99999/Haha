from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


def _sample_without_replacement(rng, population_size: int, sample_size: int) -> np.ndarray:
    if sample_size <= 0 or sample_size > population_size:
        raise ValueError("sample_size must be in [1, population_size].")

    return np.sort(rng.choice(population_size, size=sample_size, replace=False))


def _sample_from_values(rng, values: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size <= 0 or sample_size > len(values):
        raise ValueError("sample_size must be in [1, len(values)].")

    return np.sort(rng.choice(values, size=sample_size, replace=False))


def make_lira_splits(
    *,
    dataset_size: int,
    num_references: int,
    target_train_size: int,
    reference_train_size: int,
    output_dir: Union[str, Path],
    audit_size: Optional[int] = None,
    num_query_member: Optional[int] = None,
    num_query_nonmember: Optional[int] = None,
    target_train_first_n: Optional[int] = None,
    member_pool_start: Optional[int] = None,
    member_pool_end: Optional[int] = None,
    nonmember_pool_start: Optional[int] = None,
    nonmember_pool_end: Optional[int] = None,
    reference_exclude_candidates: bool = False,
    reference_exclude_target_train: bool = False,
    balanced_reference_candidates: bool = False,
    seed: int = 0,
) -> dict:
    if dataset_size <= 1:
        raise ValueError("dataset_size must be larger than 1.")

    if num_references <= 0:
        raise ValueError("num_references must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    if target_train_first_n is not None:
        if target_train_first_n != target_train_size:
            raise ValueError("target_train_first_n must equal target_train_size.")

        if target_train_first_n > dataset_size:
            raise ValueError("target_train_first_n cannot exceed dataset_size.")

        target_train_indices = np.arange(target_train_first_n, dtype=np.int64)
    else:
        target_train_indices = _sample_without_replacement(
            rng,
            dataset_size,
            target_train_size,
        )

    target_train_set = set(int(x) for x in target_train_indices)
    all_indices = np.arange(dataset_size, dtype=np.int64)
    target_mask = np.zeros(dataset_size, dtype=bool)
    target_mask[target_train_indices] = True

    if num_query_member is not None or num_query_nonmember is not None:
        if num_query_member is None or num_query_nonmember is None:
            raise ValueError(
                "num_query_member and num_query_nonmember must be provided together."
            )

        if member_pool_start is not None or member_pool_end is not None:
            if member_pool_start is None or member_pool_end is None:
                raise ValueError("member_pool_start and member_pool_end must be set together.")

            member_pool = np.arange(member_pool_start, member_pool_end, dtype=np.int64)
            member_pool = member_pool[np.isin(member_pool, target_train_indices)]
        else:
            member_pool = all_indices[target_mask]

        if nonmember_pool_start is not None or nonmember_pool_end is not None:
            if nonmember_pool_start is None or nonmember_pool_end is None:
                raise ValueError("nonmember_pool_start and nonmember_pool_end must be set together.")

            nonmember_pool = np.arange(nonmember_pool_start, nonmember_pool_end, dtype=np.int64)
            nonmember_pool = nonmember_pool[~np.isin(nonmember_pool, target_train_indices)]
        else:
            nonmember_pool = all_indices[~target_mask]

        member_indices = _sample_from_values(rng, member_pool, int(num_query_member))
        nonmember_indices = _sample_from_values(rng, nonmember_pool, int(num_query_nonmember))
        candidate_indices = np.concatenate([member_indices, nonmember_indices])
        membership_labels = np.concatenate(
            [
                np.ones(len(member_indices), dtype=np.int64),
                np.zeros(len(nonmember_indices), dtype=np.int64),
            ]
        )
        order = rng.permutation(len(candidate_indices))
        candidate_indices = candidate_indices[order]
        membership_labels = membership_labels[order]
        audit_size = int(len(candidate_indices))
    else:
        audit_size = dataset_size if audit_size is None else int(audit_size)
        candidate_indices = _sample_without_replacement(rng, dataset_size, audit_size)
        membership_labels = np.asarray(
            [int(int(x) in target_train_set) for x in candidate_indices],
            dtype=np.int64,
        )

    reference_keep = np.zeros((num_references, audit_size), dtype=bool)
    reference_paths = []
    candidate_set = set(int(x) for x in candidate_indices)

    if balanced_reference_candidates:
        if num_references % 2 != 0:
            raise ValueError(
                "balanced_reference_candidates requires an even num_references."
            )

        if reference_exclude_candidates:
            raise ValueError(
                "balanced_reference_candidates conflicts with reference_exclude_candidates."
            )

        half = num_references // 2

        for candidate_id in range(audit_size):
            start = candidate_id % num_references
            selected = (start + np.arange(half)) % num_references
            reference_keep[selected, candidate_id] = True

    for ref_id in range(num_references):
        if balanced_reference_candidates:
            required_candidates = candidate_indices[reference_keep[ref_id]]
            remaining = reference_train_size - len(required_candidates)

            if remaining < 0:
                raise ValueError(
                    "reference_train_size is smaller than the balanced candidate quota."
                )

            reference_pool_mask = np.ones(dataset_size, dtype=bool)
            reference_pool_mask[candidate_indices] = False

            if reference_exclude_target_train:
                reference_pool_mask[target_train_indices] = False

            reference_pool = all_indices[reference_pool_mask]
            sampled = _sample_from_values(rng, reference_pool, remaining)
            ref_indices = np.sort(
                np.concatenate([required_candidates, sampled]).astype(np.int64)
            )
        else:
            reference_pool_mask = np.ones(dataset_size, dtype=bool)

            if reference_exclude_candidates:
                reference_pool_mask[candidate_indices] = False

            if reference_exclude_target_train:
                reference_pool_mask[target_train_indices] = False

            reference_pool = all_indices[reference_pool_mask]
            ref_indices = _sample_from_values(rng, reference_pool, reference_train_size)

        ref_set = set(int(x) for x in ref_indices)
        reference_keep[ref_id] = [int(x) in ref_set for x in candidate_indices]

        path = output_dir / f"reference_{ref_id:03d}_train_indices.npy"
        np.save(path, ref_indices.astype(np.int64))
        reference_paths.append(str(path))

    np.save(output_dir / "candidate_indices.npy", candidate_indices.astype(np.int64))
    np.save(output_dir / "target_train_indices.npy", target_train_indices.astype(np.int64))
    np.save(output_dir / "membership_labels.npy", membership_labels.astype(np.int64))
    np.save(output_dir / "reference_keep.npy", reference_keep)

    manifest = {
        "dataset_size": int(dataset_size),
        "audit_size": int(audit_size),
        "indices_are_global": True,
        "reference_keep_semantics": (
            "reference_keep[k, i] is True iff global candidate_indices[i] "
            "is included in reference k training indices."
        ),
        "num_references": int(num_references),
        "target_train_size": int(target_train_size),
        "reference_train_size": int(reference_train_size),
        "target_train_first_n": (
            int(target_train_first_n) if target_train_first_n is not None else None
        ),
        "member_pool_start": int(member_pool_start) if member_pool_start is not None else None,
        "member_pool_end": int(member_pool_end) if member_pool_end is not None else None,
        "nonmember_pool_start": int(nonmember_pool_start) if nonmember_pool_start is not None else None,
        "nonmember_pool_end": int(nonmember_pool_end) if nonmember_pool_end is not None else None,
        "num_query_member": (
            int(num_query_member) if num_query_member is not None else int(np.sum(membership_labels == 1))
        ),
        "num_query_nonmember": (
            int(num_query_nonmember) if num_query_nonmember is not None else int(np.sum(membership_labels == 0))
        ),
        "reference_exclude_candidates": bool(reference_exclude_candidates),
        "reference_exclude_target_train": bool(reference_exclude_target_train),
        "balanced_reference_candidates": bool(balanced_reference_candidates),
        "seed": int(seed),
        "candidate_indices_path": str(output_dir / "candidate_indices.npy"),
        "target_train_indices_path": str(output_dir / "target_train_indices.npy"),
        "membership_labels_path": str(output_dir / "membership_labels.npy"),
        "reference_keep_path": str(output_dir / "reference_keep.npy"),
        "reference_train_indices_paths": reference_paths,
    }

    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create target/reference splits for Offline LiRA.")
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--num_references", type=int, required=True)
    parser.add_argument("--target_train_size", type=int, required=True)
    parser.add_argument("--reference_train_size", type=int, required=True)
    parser.add_argument("--audit_size", type=int, default=None)
    parser.add_argument("--num_query_member", type=int, default=None)
    parser.add_argument("--num_query_nonmember", type=int, default=None)
    parser.add_argument("--target_train_first_n", type=int, default=None)
    parser.add_argument("--member_pool_start", type=int, default=None)
    parser.add_argument("--member_pool_end", type=int, default=None)
    parser.add_argument("--nonmember_pool_start", type=int, default=None)
    parser.add_argument("--nonmember_pool_end", type=int, default=None)
    parser.add_argument("--reference_exclude_candidates", action="store_true")
    parser.add_argument("--reference_exclude_target_train", action="store_true")
    parser.add_argument("--balanced_reference_candidates", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    manifest = make_lira_splits(**vars(args))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
