"""Pessimistic PLD privacy filter for a realized public-coin path."""

from dataclasses import asdict, dataclass
import importlib.metadata
import math
from pathlib import Path
import sys
from typing import Dict, Optional


def _load_dp_accounting():
    try:
        from dp_accounting import privacy_accountant
        from dp_accounting.pld import privacy_loss_distribution
        return privacy_accountant, privacy_loss_distribution
    except ModuleNotFoundError:
        vendor_dir = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "realized_coin_filter"
        )
        if vendor_dir.is_dir() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        try:
            from dp_accounting import privacy_accountant
            from dp_accounting.pld import privacy_loss_distribution
            return privacy_accountant, privacy_loss_distribution
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The realized-coin PLD filter requires dp-accounting==0.5.1. "
                "Install requirements-realized-filter.txt or populate "
                "third_party/realized_coin_filter."
            ) from exc


privacy_accountant, privacy_loss_distribution = _load_dp_accounting()


@dataclass(frozen=True)
class PLDPrivacyDecision:
    token: str
    branch: str
    sigma: float
    step_index: int
    allowed: bool
    delta_upper: float
    epsilon_upper: float
    candidate_small_steps: int
    candidate_large_steps: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PessimisticRealizedCoinPLDFilter:

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        sigma_small: float,
        sigma_large: float,
        value_discretization_interval: float = 1e-4,
        log_mass_truncation_bound: float = -50.0,
        tail_mass_truncation: float = 1e-15,
    ):
        if target_epsilon < 0.0:
            raise ValueError("target_epsilon must be non-negative.")
        if not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1).")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1].")
        if sigma_small <= 0.0 or sigma_large <= 0.0:
            raise ValueError("Both noise multipliers must be positive.")
        if sigma_large < sigma_small:
            raise ValueError("sigma_large must be at least sigma_small.")
        if value_discretization_interval <= 0.0:
            raise ValueError("value_discretization_interval must be positive.")
        if tail_mass_truncation < 0.0:
            raise ValueError("tail_mass_truncation must be non-negative.")

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.sample_rate = float(sample_rate)
        self.sigma_small = float(sigma_small)
        self.sigma_large = float(sigma_large)
        self.value_discretization_interval = float(
            value_discretization_interval
        )
        self.log_mass_truncation_bound = float(log_mass_truncation_bound)
        self.tail_mass_truncation = float(tail_mass_truncation)
        self._counts = {"small": 0, "large": 0}
        self._current_pld = None
        self._pending: Optional[Dict[str, object]] = None
        self._preview_count = 0

        relation = privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE
        common = {
            "sensitivity": 1.0,
            "pessimistic_estimate": True,
            "value_discretization_interval": self.value_discretization_interval,
            "log_mass_truncation_bound": self.log_mass_truncation_bound,
            "sampling_prob": self.sample_rate,
            "use_connect_dots": True,
            "neighboring_relation": relation,
        }
        self._branch_plds = {
            "small": privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=self.sigma_small,
                **common,
            ),
            "large": privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=self.sigma_large,
                **common,
            ),
        }

    def _sigma_for(self, branch: str) -> float:
        if branch == "small":
            return self.sigma_small
        if branch == "large":
            return self.sigma_large
        raise ValueError(f"Unknown branch: {branch}")

    def preview(self, branch: str) -> PLDPrivacyDecision:
        if self._pending is not None:
            raise RuntimeError("A PLD preview is already pending.")

        sigma = self._sigma_for(branch)
        branch_pld = self._branch_plds[branch]
        if self._current_pld is None:
            candidate = branch_pld
        else:
            candidate = self._current_pld.compose(
                branch_pld,
                tail_mass_truncation=self.tail_mass_truncation,
            )

        delta_upper = float(
            candidate.get_delta_for_epsilon(self.target_epsilon)
        )
        epsilon_upper = float(
            candidate.get_epsilon_for_delta(self.target_delta)
        )
        if not math.isfinite(delta_upper) or delta_upper < 0.0:
            raise RuntimeError("PLD accountant returned a non-finite delta.")
        if math.isnan(epsilon_upper) or epsilon_upper < 0.0:
            raise RuntimeError("PLD accountant returned an invalid epsilon.")

        current_delta = self.current_delta_upper()
        if delta_upper + 1e-15 < current_delta:
            raise RuntimeError(
                "PLD delta decreased after composition; numerical state is invalid."
            )

        self._preview_count += 1
        token = (
            f"pld-step-{self.committed_steps + 1}-"
            f"preview-{self._preview_count}-{branch}"
        )
        decision = PLDPrivacyDecision(
            token=token,
            branch=branch,
            sigma=sigma,
            step_index=self.committed_steps + 1,
            allowed=delta_upper <= self.target_delta,
            delta_upper=delta_upper,
            epsilon_upper=epsilon_upper,
            candidate_small_steps=(
                self.small_steps + (1 if branch == "small" else 0)
            ),
            candidate_large_steps=(
                self.large_steps + (1 if branch == "large" else 0)
            ),
        )
        if decision.allowed:
            self._pending = {"decision": decision, "candidate": candidate}
        return decision

    def commit(self, token: str) -> None:
        pending = self._require_pending(token)
        decision = pending["decision"]
        self._current_pld = pending["candidate"]
        self._counts[decision.branch] += 1
        self._pending = None

    def cancel(self, token: str) -> None:
        self._require_pending(token)
        self._pending = None

    def _require_pending(self, token: str) -> Dict[str, object]:
        if self._pending is None:
            raise RuntimeError("There is no pending PLD preview.")
        decision = self._pending["decision"]
        if decision.token != token:
            raise RuntimeError("PLD preview token mismatch.")
        return self._pending

    @property
    def committed_steps(self) -> int:
        return self.small_steps + self.large_steps

    @property
    def small_steps(self) -> int:
        return self._counts["small"]

    @property
    def large_steps(self) -> int:
        return self._counts["large"]

    def current_delta_upper(self) -> float:
        if self._current_pld is None:
            return 0.0
        return float(
            self._current_pld.get_delta_for_epsilon(self.target_epsilon)
        )

    def current_epsilon_upper(self) -> float:
        if self._current_pld is None:
            return 0.0
        return float(
            self._current_pld.get_epsilon_for_delta(self.target_delta)
        )

    def state_dict(self) -> Dict[str, object]:
        try:
            version = importlib.metadata.version("dp-accounting")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "accountant": "pessimistic_realized_public_coin_pld_filter",
            "backend": "google-dp-accounting",
            "backend_version": version,
            "neighboring_relation": "ADD_OR_REMOVE_ONE",
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "sample_rate": self.sample_rate,
            "sigma_small": self.sigma_small,
            "sigma_large": self.sigma_large,
            "value_discretization_interval": (
                self.value_discretization_interval
            ),
            "log_mass_truncation_bound": self.log_mass_truncation_bound,
            "tail_mass_truncation": self.tail_mass_truncation,
            "pessimistic_estimate": True,
            "small_steps": self.small_steps,
            "large_steps": self.large_steps,
            "committed_steps": self.committed_steps,
            "delta_upper_at_target_epsilon": self.current_delta_upper(),
            "epsilon_upper_at_target_delta": self.current_epsilon_upper(),
        }
