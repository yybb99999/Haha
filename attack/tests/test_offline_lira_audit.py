from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from attacks.common.config import build_main_command, save_json, sha256_file
from attacks.common.logits import logit_scaled_confidence, save_logits
from attacks.common.metrics import bootstrap_attack_metrics, evaluate_attack_scores
from attacks.lira.evaluate import (
    evaluate_offline_lira,
    _validate_audit,
    _validate_non_dp_experiment_role,
    _validate_runtime_metrics,
)
from attacks.lira.score_offline_lira import offline_lira_scores


class OfflineLiRANumericTests(unittest.TestCase):
    def test_stable_logit_scaled_confidence(self):
        logits = np.asarray([[2.0, 1.0, -1.0], [1000.0, -1000.0, 0.0]])
        labels = np.asarray([0, 0])
        scores = logit_scaled_confidence(logits, labels)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[1], 900.0)

        probabilities = torch.softmax(torch.tensor(logits[:1]), dim=1).numpy()
        expected = np.log(probabilities[0, 0] / (1.0 - probabilities[0, 0]))
        self.assertAlmostEqual(scores[0], expected, places=10)

    def test_tail_score_is_stable_and_monotone(self):
        references = np.tile(np.linspace(-1.0, 1.0, 8)[:, None], (1, 3))
        result = offline_lira_scores(
            target_scores=np.asarray([0.0, 10.0, 100.0]),
            reference_scores=references,
            score_mode="paper_tail_fixed",
            min_out_references=8,
            require_all_out=True,
        )
        scores = result["attack_scores"]
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.diff(scores) > 0.0))

    def test_multi_query_statistic_matches_official_probability_average(self):
        logits = np.asarray(
            [
                [[2.0, 0.0, -1.0], [1.0, 0.5, -0.5]],
                [[0.0, 2.0, -1.0], [0.5, 1.0, -0.5]],
            ],
            dtype=np.float64,
        )
        labels = np.asarray([0, 1], dtype=np.int64)
        scores = logit_scaled_confidence(logits, labels)

        probabilities = torch.softmax(torch.tensor(logits), dim=2).numpy()
        rows = np.arange(labels.size)
        y_true = probabilities[rows, :, labels]
        expected = np.log(np.mean(y_true, axis=1)) - np.log(
            np.mean(1.0 - y_true, axis=1)
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)

    def test_official_logpdf_is_minimal_at_out_center(self):
        references = np.tile(np.linspace(-1.0, 1.0, 8)[:, None], (1, 3))
        result = offline_lira_scores(
            target_scores=np.asarray([0.0, 2.0, -2.0]),
            reference_scores=references,
            score_mode="official_logpdf_fixed",
            min_out_references=8,
            require_all_out=True,
        )
        self.assertLess(result["attack_scores"][0], result["attack_scores"][1])
        self.assertLess(result["attack_scores"][0], result["attack_scores"][2])

    def test_missing_out_references_fails(self):
        with self.assertRaises(ValueError):
            offline_lira_scores(
                target_scores=np.zeros(2),
                reference_scores=np.zeros((2, 2)),
                reference_keep=np.ones((2, 2), dtype=bool),
                min_out_references=1,
            )

    def test_metrics_and_bootstrap(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.0, 0.1, 0.9, 1.0])
        evaluation = evaluate_attack_scores(labels, scores)
        self.assertEqual(evaluation["summary"]["auc"], 1.0)
        bootstrap = bootstrap_attack_metrics(labels, scores, repeats=10, seed=3)
        self.assertEqual(bootstrap["auc"]["mean"], 1.0)


class OfflineLiRAAuditValidationTests(unittest.TestCase):
    def test_non_dp_command_uses_isolated_attack_trainer(self):
        command = build_main_command(
            "python",
            "/project",
            {
                "algorithm": "DPSGD",
                "training_mode": "non_dp",
                "accountant": "none",
                "noise_mode": "none",
            },
        )
        self.assertEqual(command[:5], ["python", "-u", "-m", "attacks.lira.train_non_dp", "--algorithm"])
        self.assertNotIn("main.py", " ".join(command))

    def test_non_dp_runtime_metrics_are_not_labeled_private(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            refs_dir = output_dir / "references"
            refs_dir.mkdir()
            metrics = {
                "train_size": 4,
                "batch_size": 2,
                "sample_rate": 0.5,
                "epsilon": None,
                "privacy_guarantee": "none",
                "accounted_steps": 0,
                "actual_dp_updates": 0,
                "training_steps": 3,
                "actual_updates": 3,
            }
            save_json(output_dir / "target_metrics.json", metrics)
            save_json(refs_dir / "reference_000_metrics.json", metrics)
            config = {
                "train": {
                    "training_mode": "non_dp",
                    "accountant": "none",
                    "batch_size": 2,
                    "target_steps": 3,
                }
            }
            split_manifest = {"target_train_size": 4}
            result = _validate_runtime_metrics(
                config,
                split_manifest,
                output_dir,
                expected_k=1,
            )
            self.assertEqual(result["training_mode"], "non_dp")
            self.assertEqual(result["target_training_steps"], 3)

    def _build_fixture(self, root: Path):
        output_dir = root / "output"
        refs_dir = output_dir / "references"
        split_dir = root / "split"
        refs_dir.mkdir(parents=True)
        split_dir.mkdir(parents=True)
        candidates = np.asarray([0, 1, 4, 5], dtype=np.int64)
        target_train = np.asarray([0, 1, 2, 3], dtype=np.int64)
        membership = np.asarray([1, 1, 0, 0], dtype=np.int64)
        reference_keep = np.zeros((2, 4), dtype=bool)
        reference_indices = np.asarray([2, 3, 6, 7], dtype=np.int64)
        np.save(split_dir / "candidate_indices.npy", candidates)
        np.save(split_dir / "target_train_indices.npy", target_train)
        np.save(split_dir / "membership_labels.npy", membership)
        np.save(split_dir / "reference_keep.npy", reference_keep)
        reference_paths = []

        for ref_id in range(2):
            path = split_dir / f"reference_{ref_id:03d}_train_indices.npy"
            np.save(path, reference_indices)
            reference_paths.append(str(path))

        split_manifest = {
            "num_references": 2,
            "target_train_size": 4,
            "reference_train_size": 4,
            "candidate_indices_path": str(split_dir / "candidate_indices.npy"),
            "target_train_indices_path": str(split_dir / "target_train_indices.npy"),
            "membership_labels_path": str(split_dir / "membership_labels.npy"),
            "reference_keep_path": str(split_dir / "reference_keep.npy"),
            "reference_train_indices_paths": reference_paths,
        }
        split_manifest_path = split_dir / "split_manifest.json"
        save_json(split_manifest_path, split_manifest)
        train = {
            "algorithm": "DPSGD",
            "dataset_name": "CIFAR-10",
            "audit_train_pool": "combined",
            "lr": 4.0,
            "momentum": 0.9,
            "batch_size": 2,
            "C_t": 0.1,
            "epsilon": 1.0,
            "delta": 1e-5,
            "accountant": "rdp",
            "noise_mode": "gaussian",
            "sigma_t": 2.0,
            "target_steps": 0,
        }
        config = {
            "output_dir": str(output_dir),
            "split_manifest_path": str(split_manifest_path),
            "train": train,
            "export": {"candidate_pool": "combined"},
            "lira": {
                "expected_k_refs": 2,
                "require_all_candidates_out": True,
                "min_out_references": 2,
            },
        }
        logits = np.asarray(
            [[2.0, 0.0], [1.5, 0.0], [0.0, 1.5], [0.0, 2.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        checkpoint_paths = [output_dir / "target_model.pth"] + [
            refs_dir / f"reference_{ref_id:03d}.pth" for ref_id in range(2)
        ]

        for index, checkpoint_path in enumerate(checkpoint_paths):
            args = dict(train)
            args["seed"] = index
            torch.save({"model_state_dict": {}, "args": args}, checkpoint_path)
            metrics = {
                "train_size": 4,
                "batch_size": 2,
                "sample_rate": 0.5,
                "epsilon": 0.9,
                "accounted_steps": 3,
                "actual_dp_updates": 3,
            }
            metrics_path = (
                output_dir / "target_metrics.json"
                if index == 0
                else refs_dir / f"reference_{index - 1:03d}_metrics.json"
            )
            save_json(metrics_path, metrics)
            logits_path = (
                output_dir / "target_logits.npz"
                if index == 0
                else refs_dir / f"reference_{index - 1:03d}_logits.npz"
            )
            save_logits(
                logits_path,
                logits,
                labels,
                extra={
                    "query_indices": candidates,
                    "checkpoint_path": np.asarray(str(checkpoint_path)),
                    "checkpoint_sha256": np.asarray(sha256_file(checkpoint_path)),
                    "indices_path": np.asarray(str(split_dir / "candidate_indices.npy")),
                    "indices_sha256": np.asarray(
                        sha256_file(split_dir / "candidate_indices.npy")
                    ),
                    "candidate_pool": np.asarray("combined"),
                },
            )

        return config, split_manifest, output_dir

    def test_non_dp_direct_comparison_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split_manifest, output_dir = self._build_fixture(Path(directory))
            config["train"]["training_mode"] = "non_dp"
            config["experiment_role"] = "attack_power_positive_control_only"
            config["direct_comparison_allowed"] = True

            with self.assertRaises(RuntimeError):
                _validate_audit(config, split_manifest, output_dir)

    def test_parameter_aligned_non_dp_role_validates_declared_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "formal_output"
            output_dir.mkdir()
            split_path = str(root / "split_manifest.json")
            reference_config_path = root / "formal.json"
            save_json(
                reference_config_path,
                {
                    "output_dir": str(output_dir),
                    "split_manifest_path": split_path,
                    "train": {
                        "algorithm": "DPSGD",
                        "dataset_name": "CIFAR-10",
                        "lr": 4.0,
                        "momentum": 0.9,
                        "batch_size": 8192,
                        "audit_train_pool": "combined",
                    },
                },
            )
            save_json(output_dir / "target_metrics.json", {"accounted_steps": 516})
            config = {
                "experiment_role": "parameter_aligned_non_dp_baseline",
                "direct_comparison_allowed": True,
                "split_manifest_path": split_path,
                "comparison_alignment": {
                    "reference_configs": [str(reference_config_path)],
                    "max_step_difference": 0,
                    "expected_train_values": {
                        "lr": 4.0,
                        "batch_size": 8192,
                        "sampling_scheme": "poisson",
                    },
                },
                "train": {
                    "training_mode": "non_dp",
                    "algorithm": "DPSGD",
                    "dataset_name": "CIFAR-10",
                    "lr": 4.0,
                    "momentum": 0.9,
                    "batch_size": 8192,
                    "target_steps": 516,
                    "sampling_scheme": "poisson",
                    "audit_train_pool": "combined",
                },
            }
            result = _validate_non_dp_experiment_role(config)
            self.assertEqual(len(result["validated_reference_configs"]), 1)

    def test_parameter_aligned_non_dp_role_rejects_mismatch(self):
        config = {
            "experiment_role": "parameter_aligned_non_dp_baseline",
            "direct_comparison_allowed": True,
            "comparison_alignment": {
                "expected_train_values": {"lr": 4.0}
            },
            "train": {
                "training_mode": "non_dp",
                "lr": 0.1,
            },
        }

        with self.assertRaises(RuntimeError):
            _validate_non_dp_experiment_role(config)

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split_manifest, output_dir = self._build_fixture(Path(directory))
            audit = _validate_audit(config, split_manifest, output_dir)
            self.assertEqual(audit["validation"]["status"], "passed")

    def test_multi_query_logits_subdir_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split_manifest, output_dir = self._build_fixture(Path(directory))
            logits_dir = output_dir / "paper_queries_2"
            logits_refs_dir = logits_dir / "references"
            logits_refs_dir.mkdir(parents=True)
            source_paths = [output_dir / "target_logits.npz"] + [
                output_dir / "references" / f"reference_{ref_id:03d}_logits.npz"
                for ref_id in range(2)
            ]
            destination_paths = [logits_dir / "target_logits.npz"] + [
                logits_refs_dir / f"reference_{ref_id:03d}_logits.npz"
                for ref_id in range(2)
            ]

            for source_path, destination_path in zip(
                source_paths, destination_paths
            ):
                with np.load(source_path, allow_pickle=False) as source:
                    payload = {key: source[key] for key in source.files}
                logits = np.stack([payload.pop("logits")] * 2, axis=1)
                labels = payload.pop("labels")
                payload["query_augmentations"] = np.asarray(
                    ["identity", "hflip"]
                )
                save_logits(destination_path, logits, labels, extra=payload)

            config["export"] = {
                "candidate_pool": "combined",
                "logits_subdir": "paper_queries_2",
                "query_augmentations": ["identity", "hflip"],
            }
            config["lira"]["results_subdir"] = "results_paper_queries_2"
            audit = _validate_audit(config, split_manifest, output_dir)
            self.assertEqual(
                audit["validation"]["query_augmentations"],
                ["identity", "hflip"],
            )

            config["lira"]["score_modes"] = ["official_logpdf_fixed"]
            config["lira"]["bootstrap_repeats"] = 2
            config_path = Path(directory) / "paper_queries_2.json"
            save_json(config_path, config)
            evaluate_offline_lira(str(config_path))
            self.assertTrue(
                (
                    output_dir
                    / "results_paper_queries_2"
                    / "official_logpdf_fixed"
                    / "metrics.json"
                ).is_file()
            )

    def test_missing_reference_logits_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config, split_manifest, output_dir = self._build_fixture(Path(directory))
            (output_dir / "references" / "reference_001_logits.npz").unlink()

            with self.assertRaises(RuntimeError):
                _validate_audit(config, split_manifest, output_dir)

    def test_strict_json_replaces_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.json"
            save_json(path, {"bad": float("inf"), "good": 1.0})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(payload["bad"])


if __name__ == "__main__":
    unittest.main()
