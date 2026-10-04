from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


VERSION = "dpbisgd_three_stage_parameter_impact_v2_cli_names"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRYPOINT = PROJECT_ROOT / "tools" / "run_dpbisgd_realized_coin_filter.py"



PROFILES = {
    "cifar10-dpbisgd": {
        "algorithm": "DP-BiSGD",
        "dataset_name": "CIFAR-10",
        "dataset_size": 50_000,
        "batch_size": 8_192,
        "C_t": 0.1,
        "lr": 4.0,
        "momentum": 0.9,
    },
    "mnist-dpbisgd": {
        "algorithm": "DP-BiSGD",
        "dataset_name": "MNIST",
        "dataset_size": 60_000,
        "batch_size": 1_024,
        "C_t": 0.1,
        "lr": 2.0,
        "momentum": 0.9,
    },
    "fmnist-dpbisgd": {
        "algorithm": "DP-BiSGD",
        "dataset_name": "FMNIST",
        "dataset_size": 60_000,
        "batch_size": 2_048,
        "C_t": 0.1,
        "lr": 4.0,
        "momentum": 0.9,
    },
}

PROFILE_ALIASES = {
    "cifar10-dpsgd": "cifar10-dpbisgd",
    "mnist-dpsgd": "mnist-dpbisgd",
    "fmnist-dpsgd": "fmnist-dpbisgd",
}


CSV_FIELDS = [
    "Stage",
    "Selected",
    "Profile",
    "Dataset",
    "Method",
    "Epsilon",
    "Delta",
    "Sigma_Base",
    "Sigma_Small_Ratio",
    "Sigma_Small",
    "Sigma_Large_Multiplier",
    "Sigma_Large",
    "P_Large",
    "Best_Acc",
    "Fin_Acc",
    "Best_Acc_t",
    "Fin_Acc_T",
    "T",
    "Small_Steps",
    "Large_Steps",
    "Observed_P_Large",
    "Epsilon_Upper_Final",
    "Delta_Upper_Final",
    "Rejected_Next_Delta_Upper",
    "Stop_Reason",
    "Seed",
    "Branch_Seed",
    "Predicted_T",
    "Elapsed_Seconds",
    "Run_Path",
]


@dataclass(frozen=True)
class Candidate:
    sigma_small: float
    sigma_large: float
    p_large: float

    @property
    def key(self) -> str:
        return (
            f"ss-{_number_slug(self.sigma_small)}_"
            f"sl-{_number_slug(self.sigma_large)}_"
            f"p-{_number_slug(self.p_large)}"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: float) -> str:
    return format(float(value), ".12g")


def _number_slug(value: float) -> str:
    return _number(value).replace("-", "m").replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _same_float(left, right, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _candidate_values(values: Iterable[float]) -> List[float]:
    result = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("All sigma values and multipliers must be positive and finite.")
        if not any(_same_float(value, existing) for existing in result):
            result.append(value)
    if not result:
        raise ValueError("A candidate grid cannot be empty.")
    return result


def _probability_values(values: Iterable[float]) -> List[float]:
    result = _candidate_values(values)
    if any(value >= 1.0 for value in result):
        raise ValueError("Every p_large candidate must be in (0, 1).")
    return result


class ImpactSearch:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.profile_cli = args.profile
        self.profile_name = PROFILE_ALIASES.get(args.profile, args.profile)
        self.profile = dict(PROFILES[self.profile_name])
        self.output_dir = args.output_dir.resolve()
        self.runs_dir = self.output_dir / "runs"
        self.logs_dir = self.output_dir / "logs"
        self.states_dir = self.output_dir / "states"
        self.plans_dir = self.output_dir / "plans"
        self.python = args.python.resolve()
        self.sigma_base = float(args.sigma_base)
        self.small_ratios = _candidate_values(args.small_ratios)
        self.large_multipliers = _candidate_values(args.large_multipliers)
        self.p_values = _probability_values(args.p_values)
        self.results: Dict[str, Mapping[str, object]] = {}
        self.stage_rows: List[Mapping[str, object]] = []
        self.stage_summary: Dict[str, object] = {}

        if not math.isfinite(self.sigma_base) or self.sigma_base <= 0.0:
            raise ValueError("sigma_base must be positive and finite.")
        if not 0.0 < args.initial_p_large < 1.0:
            raise ValueError("initial_p_large must be in (0, 1).")
        if args.epsilon < 0.0 or not 0.0 < args.delta < 1.0:
            raise ValueError("epsilon/delta are invalid.")
        if args.workers <= 0:
            raise ValueError("workers must be positive.")
        if args.preflight_hard_cap <= 0:
            raise ValueError("preflight_hard_cap must be positive.")
        if (
            not math.isfinite(args.initial_sigma_large_multiplier)
            or args.initial_sigma_large_multiplier <= 0.0
        ):
            raise ValueError("initial_sigma_large_multiplier must be positive.")
        if args.mode in ("stage2", "stage3"):
            if args.fixed_sigma_small is None or not math.isfinite(args.fixed_sigma_small):
                raise ValueError(f"--fixed_sigma_small is required for --mode {args.mode}.")
            if args.fixed_sigma_small <= 0.0:
                raise ValueError("fixed_sigma_small must be positive.")
        if args.mode == "stage3":
            if args.fixed_sigma_large is None or not math.isfinite(args.fixed_sigma_large):
                raise ValueError("--fixed_sigma_large is required for --mode stage3.")
            if args.fixed_sigma_large < args.fixed_sigma_small:
                raise ValueError("fixed_sigma_large must be at least fixed_sigma_small.")
        if args.mode in ("all", "stage1") and (
            args.initial_sigma_large_multiplier < max(self.small_ratios)
        ):
            raise ValueError(
                "The initial sigma_large multiplier must be at least every "
                "sigma_small ratio."
            )
        if args.mode == "stage2" and (
            min(self.large_multipliers) * self.sigma_base < args.fixed_sigma_small
        ):
            raise ValueError(
                "At least one stage-2 sigma_large candidate is below fixed_sigma_small."
            )

    def protocol(self) -> Mapping[str, object]:
        return {
            "version": VERSION,
            "mode": self.args.mode,
            "profile_name": self.profile_name,
            "profile_cli": self.profile_cli,
            "profile": self.profile,
            "method": self.profile["algorithm"],
            "epsilon": self.args.epsilon,
            "delta": self.args.delta,
            "sigma_base": self.sigma_base,
            "small_ratios": self.small_ratios,
            "large_multipliers": self.large_multipliers,
            "p_values": self.p_values,
            "initial_sigma_large_multiplier": self.args.initial_sigma_large_multiplier,
            "initial_p_large": self.args.initial_p_large,
            "fixed_sigma_small": self.args.fixed_sigma_small,
            "fixed_sigma_large": self.args.fixed_sigma_large,
            "large_step_update": self.args.large_step_update,
            "large_step_lr_scale": self.args.large_step_lr_scale,
            "seed": self.args.seed,
            "branch_seed": self.args.branch_seed,
            "device": self.args.device,
            "pld_discretization": self.args.pld_discretization,
            "pld_log_mass_truncation": self.args.pld_log_mass_truncation,
            "pld_tail_mass_truncation": self.args.pld_tail_mass_truncation,
            "preflight_hard_cap": self.args.preflight_hard_cap,
            "selection_metric": "Fin_Acc, with Best_Acc as deterministic tie-breaker",
            "seed_policy": "single fixed seed for fast impact screening",
            "training_entrypoint": str(TRAIN_ENTRYPOINT),
            "training_entrypoint_sha256": _sha256(TRAIN_ENTRYPOINT),
            "orchestrator_sha256": _sha256(Path(__file__).resolve()),
            "python": str(self.python),
        }

    def prepare(self) -> None:
        if not TRAIN_ENTRYPOINT.is_file():
            raise FileNotFoundError(
                f"Missing realized-coin training entrypoint: {TRAIN_ENTRYPOINT}"
            )
        if not self.python.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {self.python}")

        expected = self.protocol()
        protocol_path = self.output_dir / "protocol.json"
        if self.output_dir.exists():
            if not self.args.resume:
                raise FileExistsError(
                    f"Output directory already exists; pass --resume to audit and reuse "
                    f"completed candidates: {self.output_dir}"
                )
            if not protocol_path.is_file():
                raise RuntimeError("Existing output has no protocol.json; refusing to reuse it.")
            if _read_json(protocol_path) != expected:
                raise RuntimeError("Existing protocol differs from this invocation.")
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            _atomic_json(protocol_path, expected)

        for directory in (
            self.runs_dir,
            self.logs_dir,
            self.states_dir,
            self.plans_dir,
        ):
            directory.mkdir(exist_ok=True)

    def predict_horizon(self, candidate: Candidate) -> Mapping[str, object]:
        if candidate.sigma_large < candidate.sigma_small:
            raise ValueError(
                f"sigma_large must be >= sigma_small for candidate {candidate.key}."
            )

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from privacy_analysis.filters import PessimisticRealizedCoinPLDFilter
        from utils.realized_coin_sampling import RealizedCoinSampler

        sample_rate = self.profile["batch_size"] / self.profile["dataset_size"]
        privacy_filter = PessimisticRealizedCoinPLDFilter(
            target_epsilon=self.args.epsilon,
            target_delta=self.args.delta,
            sample_rate=sample_rate,
            sigma_small=candidate.sigma_small,
            sigma_large=candidate.sigma_large,
            value_discretization_interval=self.args.pld_discretization,
            log_mass_truncation_bound=self.args.pld_log_mass_truncation,
            tail_mass_truncation=self.args.pld_tail_mass_truncation,
        )
        branch_sampler = RealizedCoinSampler(
            p_large=candidate.p_large,
            sigma_small=candidate.sigma_small,
            sigma_large=candidate.sigma_large,
            seed=self.args.branch_seed,
        )

        rejected = None
        while privacy_filter.committed_steps < self.args.preflight_hard_cap:
            coin = branch_sampler.draw()
            decision = privacy_filter.preview(coin.branch)
            if not decision.allowed:
                rejected = decision
                break
            privacy_filter.commit(decision.token)
        if rejected is None:
            raise RuntimeError(
                f"Candidate {candidate.key} did not exhaust the budget before the "
                f"preflight hard cap ({self.args.preflight_hard_cap})."
            )

        return {
            "candidate": asdict(candidate),
            "candidate_key": candidate.key,
            "predicted_T": privacy_filter.committed_steps,
            "predicted_small_steps": privacy_filter.small_steps,
            "predicted_large_steps": privacy_filter.large_steps,
            "rejected_next_branch": rejected.branch,
            "rejected_next_delta_upper": rejected.delta_upper,
            "rejected_next_epsilon_upper": rejected.epsilon_upper,
            "max_updates_safety": privacy_filter.committed_steps + 1,
            "sample_rate": sample_rate,
        }

    def command(self, candidate: Candidate, plan: Mapping[str, object]) -> List[str]:
        run_dir = self.runs_dir / candidate.key
        command = [
            str(self.python),
            "-B",
            "-u",
            str(TRAIN_ENTRYPOINT),
            "--algorithm",
            self.profile["algorithm"],
            "--dataset_name",
            self.profile["dataset_name"],
            "--epsilon",
            _number(self.args.epsilon),
            "--delta",
            _number(self.args.delta),
            "--sigma_small",
            _number(candidate.sigma_small),
            "--sigma_large",
            _number(candidate.sigma_large),
            "--p_large",
            _number(candidate.p_large),
            "--batch_size",
            str(self.profile["batch_size"]),
            "--C_t",
            _number(self.profile["C_t"]),
            "--lr",
            _number(self.profile["lr"]),
            "--momentum",
            _number(self.profile["momentum"]),
            "--large_step_update",
            self.args.large_step_update,
            "--large_step_lr_scale",
            _number(self.args.large_step_lr_scale),
            "--max_updates_safety",
            str(plan["max_updates_safety"]),
            "--seed",
            str(self.args.seed),
            "--branch_seed",
            str(self.args.branch_seed),
            "--device",
            self.args.device,
            "--pld_discretization",
            _number(self.args.pld_discretization),
            "--pld_log_mass_truncation",
            _number(self.args.pld_log_mass_truncation),
            "--pld_tail_mass_truncation",
            _number(self.args.pld_tail_mass_truncation),
            "--output_dir",
            str(run_dir),
            "--run_tag",
            candidate.key,
        ]
        return command

    def _write_plan(self, candidate: Candidate) -> Mapping[str, object]:
        path = self.plans_dir / f"{candidate.key}.json"
        plan = self.predict_horizon(candidate)
        plan = dict(plan, command=self.command(candidate, plan))
        if path.exists():
            if _read_json(path) != plan:
                raise RuntimeError(f"Existing candidate plan changed: {path}")
        else:
            _atomic_json(path, plan)
        return plan

    def _validate_completed(
        self,
        candidate: Candidate,
        plan: Mapping[str, object],
    ) -> Mapping[str, object]:
        run_dir = self.runs_dir / candidate.key
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                f"Partial output exists for {candidate.key}; refusing to restart or overwrite it."
            )

        metrics = _read_json(metrics_path)
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "passed":
            raise RuntimeError(f"Manifest did not pass for {candidate.key}.")
        for name, digest in manifest.get("artifacts", {}).items():
            artifact = run_dir / name
            if not artifact.is_file() or _sha256(artifact) != digest:
                raise RuntimeError(f"Artifact hash mismatch: {artifact}")

        expected_values = {
            "algorithm": self.profile["algorithm"],
            "dataset_name": self.profile["dataset_name"],
            "epsilon_target": self.args.epsilon,
            "delta_target": self.args.delta,
            "sigma_small": candidate.sigma_small,
            "sigma_large": candidate.sigma_large,
            "p_large": candidate.p_large,
            "batch_size": self.profile["batch_size"],
            "C_t": self.profile["C_t"],
            "lr": self.profile["lr"],
            "momentum": self.profile["momentum"],
            "large_step_lr_scale": self.args.large_step_lr_scale,
            "seed": self.args.seed,
            "branch_seed": self.args.branch_seed,
            "max_updates_safety": plan["max_updates_safety"],
        }
        for name, expected in expected_values.items():
            actual = metrics.get(name)
            if isinstance(expected, float):
                if actual is None or not _same_float(actual, expected):
                    raise RuntimeError(
                        f"Metric/config mismatch for {candidate.key}: "
                        f"{name}={actual!r}, expected {expected!r}."
                    )
            elif actual != expected:
                raise RuntimeError(
                    f"Metric/config mismatch for {candidate.key}: "
                    f"{name}={actual!r}, expected {expected!r}."
                )

        required_equal_steps = [
            metrics.get("T_final"),
            metrics.get("final_iter"),
            metrics.get("accounted_steps"),
            metrics.get("actual_dp_updates"),
            metrics.get("stepwise_total_steps"),
            plan["predicted_T"],
        ]
        if len(set(required_equal_steps)) != 1:
            raise RuntimeError(
                f"Accounted, executed, final, and predicted steps differ for {candidate.key}."
            )
        total = int(metrics["T_final"])
        small = int(metrics["stepwise_small_steps"])
        large = int(metrics["stepwise_large_steps"])
        if small + large != total:
            raise RuntimeError(f"Branch counts do not sum to T for {candidate.key}.")
        if small != plan["predicted_small_steps"] or large != plan["predicted_large_steps"]:
            raise RuntimeError(f"Training branch path differs from preflight for {candidate.key}.")

        for name in (
            "best_acc",
            "final_acc",
            "observed_p_large",
            "epsilon_upper_final",
            "delta_upper_final_at_target_epsilon",
            "rejected_next_delta_upper",
        ):
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite {name} for {candidate.key}.")
        if metrics.get("status") != "passed" or metrics.get("privacy_status") != "passed":
            raise RuntimeError(f"Run status did not pass for {candidate.key}.")
        if metrics.get("stop_reason") != "privacy_budget":
            raise RuntimeError(f"Safety cap bound the run for {candidate.key}.")
        if not metrics.get("stop_before_exceed"):
            raise RuntimeError(f"Missing stop-before-exceed proof for {candidate.key}.")
        if float(metrics["delta_upper_final_at_target_epsilon"]) > self.args.delta + 1e-15:
            raise RuntimeError(f"Final PLD delta exceeds the target for {candidate.key}.")
        if float(metrics["epsilon_upper_final"]) > self.args.epsilon + 1e-12:
            raise RuntimeError(f"Final PLD epsilon exceeds the target for {candidate.key}.")
        if float(metrics["rejected_next_delta_upper"]) <= self.args.delta:
            raise RuntimeError(f"Rejected next step did not exceed the target for {candidate.key}.")
        if not 0.0 <= float(metrics["final_acc"]) <= float(metrics["best_acc"]) <= 100.0:
            raise RuntimeError(f"Accuracy ordering is invalid for {candidate.key}.")
        source_hashes = metrics.get("source_sha256", {})
        relative_runner = "tools/run_dpbisgd_realized_coin_filter.py"
        if source_hashes.get(relative_runner) != _sha256(TRAIN_ENTRYPOINT):
            raise RuntimeError(f"Training entrypoint hash mismatch for {candidate.key}.")

        return metrics

    def _run_candidate(
        self,
        candidate: Candidate,
        plan: Mapping[str, object],
    ) -> Mapping[str, object]:
        run_dir = self.runs_dir / candidate.key
        stdout_path = self.logs_dir / f"{candidate.key}.stdout.log"
        stderr_path = self.logs_dir / f"{candidate.key}.stderr.log"
        state_path = self.states_dir / f"{candidate.key}.json"

        if run_dir.exists():
            metrics = self._validate_completed(candidate, plan)
            return dict(metrics, elapsed_seconds=None, reused=True)
        if stdout_path.exists() or stderr_path.exists() or state_path.exists():
            raise RuntimeError(
                f"Partial orchestration artifacts exist for {candidate.key}; refusing to restart."
            )

        command = self.command(candidate, plan)
        state = {
            "status": "starting",
            "candidate": asdict(candidate),
            "candidate_key": candidate.key,
            "started_at": _now(),
            "command": command,
        }
        _atomic_json(state_path, state)
        started = time.monotonic()
        try:
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONHASHSEED": str(self.args.seed),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
                "x", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                state.update(status="running", pid=process.pid)
                _atomic_json(state_path, state)
                return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"Training exited with code {return_code}; see {stderr_path}."
                )
            elapsed = time.monotonic() - started
            metrics = self._validate_completed(candidate, plan)
            state.update(
                status="passed",
                exit_code=0,
                ended_at=_now(),
                elapsed_seconds=elapsed,
                metrics_sha256=_sha256(run_dir / "metrics.json"),
                manifest_sha256=_sha256(run_dir / "manifest.json"),
                stdout_sha256=_sha256(stdout_path),
                stderr_sha256=_sha256(stderr_path),
            )
            _atomic_json(state_path, state)
            return dict(metrics, elapsed_seconds=elapsed, reused=False)
        except BaseException:
            state.update(
                status="failed",
                ended_at=_now(),
                elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(),
            )
            _atomic_json(state_path, state)
            raise

    @staticmethod
    def _select(candidates: Sequence[Candidate], results: Mapping[str, Mapping[str, object]]) -> Candidate:
        indexed = list(enumerate(candidates))
        _, selected = max(
            indexed,
            key=lambda item: (
                float(results[item[1].key]["final_acc"]),
                float(results[item[1].key]["best_acc"]),
                -item[0],
            ),
        )
        return selected

    def run_stage(self, name: str, candidates: Sequence[Candidate]) -> Candidate:
        unique = []
        for candidate in candidates:
            if candidate.key not in {item.key for item in unique}:
                unique.append(candidate)

        plans = {candidate.key: self._write_plan(candidate) for candidate in unique}
        pending = [candidate for candidate in unique if candidate.key not in self.results]
        errors = []
        with ThreadPoolExecutor(max_workers=min(self.args.workers, len(pending) or 1)) as pool:
            futures = {
                pool.submit(self._run_candidate, candidate, plans[candidate.key]): candidate
                for candidate in pending
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    self.results[candidate.key] = future.result()
                    metrics = self.results[candidate.key]
                    print(
                        json.dumps(
                            {
                                "event": "candidate_passed",
                                "stage": name,
                                "candidate": asdict(candidate),
                                "T": metrics["T_final"],
                                "Best_Acc": metrics["best_acc"],
                                "Fin_Acc": metrics["final_acc"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except BaseException as exc:
                    errors.append((candidate, exc))
        if errors:
            joined = "; ".join(f"{c.key}: {error}" for c, error in errors)
            raise RuntimeError(f"One or more candidates failed in {name}: {joined}")

        selected = self._select(unique, self.results)
        self.stage_summary[name] = {
            "candidates": [asdict(candidate) for candidate in unique],
            "selected": asdict(selected),
            "selection_metric": "Fin_Acc",
        }
        for candidate in unique:
            self.stage_rows.append(
                self._csv_row(
                    stage=name,
                    candidate=candidate,
                    selected=candidate.key == selected.key,
                    plan=plans[candidate.key],
                    metrics=self.results[candidate.key],
                )
            )
        _atomic_json(self.output_dir / "stage_summary.json", self.stage_summary)
        self.write_csv()
        return selected

    def _csv_row(
        self,
        stage: str,
        candidate: Candidate,
        selected: bool,
        plan: Mapping[str, object],
        metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        state_path = self.states_dir / f"{candidate.key}.json"
        elapsed = None
        if state_path.is_file():
            elapsed = _read_json(state_path).get("elapsed_seconds")
        return {
            "Stage": stage,
            "Selected": selected,
            "Profile": self.profile_name,
            "Dataset": self.profile["dataset_name"],
            "Method": "DP-BiSGD",
            "Epsilon": self.args.epsilon,
            "Delta": self.args.delta,
            "Sigma_Base": self.sigma_base,
            "Sigma_Small_Ratio": candidate.sigma_small / self.sigma_base,
            "Sigma_Small": candidate.sigma_small,
            "Sigma_Large_Multiplier": candidate.sigma_large / self.sigma_base,
            "Sigma_Large": candidate.sigma_large,
            "P_Large": candidate.p_large,
            "Best_Acc": metrics["best_acc"],
            "Fin_Acc": metrics["final_acc"],
            "Best_Acc_t": metrics["best_iter"],
            "Fin_Acc_T": metrics["final_iter"],
            "T": metrics["T_final"],
            "Small_Steps": metrics["stepwise_small_steps"],
            "Large_Steps": metrics["stepwise_large_steps"],
            "Observed_P_Large": metrics["observed_p_large"],
            "Epsilon_Upper_Final": metrics["epsilon_upper_final"],
            "Delta_Upper_Final": metrics["delta_upper_final_at_target_epsilon"],
            "Rejected_Next_Delta_Upper": metrics["rejected_next_delta_upper"],
            "Stop_Reason": metrics["stop_reason"],
            "Seed": metrics["seed"],
            "Branch_Seed": metrics["branch_seed"],
            "Predicted_T": plan["predicted_T"],
            "Elapsed_Seconds": elapsed,
            "Run_Path": str(self.runs_dir / candidate.key),
        }

    def write_csv(self) -> None:
        path = self.output_dir / "impact_results.csv"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.stage_rows)
        os.replace(temporary, path)

    def run(self) -> Mapping[str, object]:
        self.prepare()
        state = {
            "status": "running",
            "started_at": _now(),
            "pid": os.getpid(),
            "version": VERSION,
        }
        _atomic_json(self.output_dir / "search.state.json", state)
        started = time.monotonic()
        try:
            initial_large = self.args.initial_sigma_large_multiplier * self.sigma_base
            if self.args.mode in ("all", "stage1"):
                stage1 = [
                    Candidate(
                        sigma_small=ratio * self.sigma_base,
                        sigma_large=initial_large,
                        p_large=self.args.initial_p_large,
                    )
                    for ratio in self.small_ratios
                ]
                best_small = self.run_stage("1_sigma_small", stage1)
            else:
                best_small = None

            if self.args.mode == "stage1":
                best = best_small
                result_scope = "local_stage1_subset"
            elif self.args.mode in ("all", "stage2"):
                sigma_small = (
                    best_small.sigma_small
                    if best_small is not None
                    else self.args.fixed_sigma_small
                )
                if sigma_small is None:
                    raise ValueError("--fixed_sigma_small is required for --mode stage2.")
                stage2 = [
                    Candidate(
                        sigma_small=float(sigma_small),
                        sigma_large=multiplier * self.sigma_base,
                        p_large=self.args.initial_p_large,
                    )
                    for multiplier in self.large_multipliers
                ]
                best_large = self.run_stage("2_sigma_large", stage2)
                if self.args.mode == "stage2":
                    best = best_large
                    result_scope = "local_stage2_subset"
            else:
                best_large = None

            if self.args.mode in ("all", "stage3"):
                sigma_small = (
                    best_large.sigma_small
                    if best_large is not None
                    else self.args.fixed_sigma_small
                )
                sigma_large = (
                    best_large.sigma_large
                    if best_large is not None
                    else self.args.fixed_sigma_large
                )
                if sigma_small is None or sigma_large is None:
                    raise ValueError(
                        "--fixed_sigma_small and --fixed_sigma_large are required "
                        "for --mode stage3."
                    )
                stage3 = [
                    Candidate(
                        sigma_small=float(sigma_small),
                        sigma_large=float(sigma_large),
                        p_large=value,
                    )
                    for value in self.p_values
                ]
                best = self.run_stage("3_p_large", stage3)
                result_scope = (
                    "full_three_stage_search"
                    if self.args.mode == "all"
                    else "local_stage3_subset"
                )

            if best is None:
                raise RuntimeError("No candidate was selected.")
            best_metrics = self.results[best.key]
            selected = {
                "status": "passed",
                "mode": self.args.mode,
                "scope": result_scope,
                "profile": self.profile_name,
                "profile_cli": self.profile_cli,
                "method": self.profile["algorithm"],
                "epsilon": self.args.epsilon,
                "delta": self.args.delta,
                "sigma_base": self.sigma_base,
                "sigma_small": best.sigma_small,
                "sigma_large": best.sigma_large,
                "p_large": best.p_large,
                "Best_Acc": best_metrics["best_acc"],
                "Fin_Acc": best_metrics["final_acc"],
                "Best_Acc_t": best_metrics["best_iter"],
                "T": best_metrics["T_final"],
                "selection_note": "Single-seed impact screening result; not a multi-seed estimate.",
            }
            _atomic_json(self.output_dir / "selected_parameters.json", selected)
            summary_manifest = {
                "status": "passed",
                "completed_at": _now(),
                "selected_parameters_sha256": _sha256(
                    self.output_dir / "selected_parameters.json"
                ),
                "impact_results_csv_sha256": _sha256(
                    self.output_dir / "impact_results.csv"
                ),
                "stage_summary_sha256": _sha256(
                    self.output_dir / "stage_summary.json"
                ),
                "completed_unique_runs": len(self.results),
            }
            _atomic_json(self.output_dir / "summary_manifest.json", summary_manifest)
            state.update(
                status="passed",
                exit_code=0,
                ended_at=_now(),
                elapsed_seconds=time.monotonic() - started,
                selected=selected,
            )
            return selected
        except BaseException:
            state.update(
                status="failed",
                exit_code=1,
                ended_at=_now(),
                elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(),
            )
            raise
        finally:
            _atomic_json(self.output_dir / "search.exit.json", state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast, single-seed, three-stage DP-BiSGD parameter-impact "
            "search from one Gaussian sigma_base reference value."
        )
    )
    parser.add_argument("--sigma_base", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help=(
            "Run all stages on one host, or run one stage with a candidate subset "
            "for cross-host distribution."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(set(PROFILES) | set(PROFILE_ALIASES)),
        default="cifar10-dpbisgd",
        help=(
            "Public experiment profile. Legacy *-dpsgd names are accepted as "
            "backward-compatible aliases."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--small_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--large_multipliers",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0, 5.0],
    )
    parser.add_argument(
        "--p_values",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument("--initial_sigma_large_multiplier", type=float, default=4.0)
    parser.add_argument("--initial_p_large", type=float, default=0.05)
    parser.add_argument("--fixed_sigma_small", type=float, default=None)
    parser.add_argument("--fixed_sigma_large", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--branch_seed", type=int, default=21260819)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--large_step_update",
        choices=["normal", "sgd_bypass", "sgd_bypass_scaled", "skip"],
        default="sgd_bypass_scaled",
    )
    parser.add_argument("--large_step_lr_scale", type=float, default=0.1)
    parser.add_argument("--pld_discretization", type=float, default=1e-4)
    parser.add_argument("--pld_log_mass_truncation", type=float, default=-50.0)
    parser.add_argument("--pld_tail_mass_truncation", type=float, default=1e-15)
    parser.add_argument("--preflight_hard_cap", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    selected = ImpactSearch(args).run()
    print(json.dumps({"event": "search_passed", **selected}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
