"""Incremental RDP reference accountant for a realized sigma sequence."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps


DEFAULT_RDP_ORDERS: Tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(11, 64))
    + [128, 256, 512]
)


@dataclass(frozen=True)
class RDPPrivacyDecision:
    token: str
    branch: str
    sigma: float
    step_index: int
    epsilon_upper: float
    best_order: float


class RealizedCoinRDPFilter:
    """Compose the RDP cost of each branch selected by the public coin."""

    def __init__(
        self,
        sample_rate: float,
        sigma_small: float,
        sigma_large: float,
        target_delta: float,
        orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    ):
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1].")
        if sigma_small <= 0.0 or sigma_large <= 0.0:
            raise ValueError("Both noise multipliers must be positive.")
        if not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1).")

        self.sample_rate = float(sample_rate)
        self.sigma_small = float(sigma_small)
        self.sigma_large = float(sigma_large)
        self.target_delta = float(target_delta)
        self.orders = np.asarray(tuple(orders), dtype=float)
        self._step_rdp = {
            "small": np.asarray(
                compute_rdp(self.sample_rate, self.sigma_small, 1, self.orders),
                dtype=float,
            ),
            "large": np.asarray(
                compute_rdp(self.sample_rate, self.sigma_large, 1, self.orders),
                dtype=float,
            ),
        }
        self._rdp = np.zeros_like(self.orders)
        self._counts = {"small": 0, "large": 0}
        self._pending: Optional[Dict[str, object]] = None

    def _sigma_for(self, branch: str) -> float:
        if branch == "small":
            return self.sigma_small
        if branch == "large":
            return self.sigma_large
        raise ValueError(f"Unknown branch: {branch}")

    def preview(self, branch: str, token: str) -> RDPPrivacyDecision:
        if self._pending is not None:
            raise RuntimeError("An RDP preview is already pending.")
        sigma = self._sigma_for(branch)
        candidate = self._rdp + self._step_rdp[branch]
        epsilon, best_order = compute_eps(
            self.orders,
            candidate,
            self.target_delta,
        )
        decision = RDPPrivacyDecision(
            token=str(token),
            branch=branch,
            sigma=sigma,
            step_index=self.committed_steps + 1,
            epsilon_upper=float(epsilon),
            best_order=float(best_order),
        )
        self._pending = {"decision": decision, "candidate": candidate}
        return decision

    def commit(self, token: str) -> None:
        pending = self._require_pending(token)
        decision = pending["decision"]
        self._rdp = np.asarray(pending["candidate"], dtype=float)
        self._counts[decision.branch] += 1
        self._pending = None

    def cancel(self, token: str) -> None:
        self._require_pending(token)
        self._pending = None

    def _require_pending(self, token: str) -> Dict[str, object]:
        if self._pending is None:
            raise RuntimeError("There is no pending RDP preview.")
        decision = self._pending["decision"]
        if decision.token != token:
            raise RuntimeError("RDP preview token mismatch.")
        return self._pending

    @property
    def committed_steps(self) -> int:
        return self._counts["small"] + self._counts["large"]

    @property
    def small_steps(self) -> int:
        return self._counts["small"]

    @property
    def large_steps(self) -> int:
        return self._counts["large"]

    def current_epsilon(self) -> Tuple[float, float]:
        if self.committed_steps == 0:
            return 0.0, float(self.orders[0])
        epsilon, best_order = compute_eps(
            self.orders,
            self._rdp,
            self.target_delta,
        )
        return float(epsilon), float(best_order)

    def state_dict(self) -> Dict[str, object]:
        epsilon, best_order = self.current_epsilon()
        return {
            "sample_rate": self.sample_rate,
            "sigma_small": self.sigma_small,
            "sigma_large": self.sigma_large,
            "target_delta": self.target_delta,
            "orders": self.orders.tolist(),
            "rdp": self._rdp.tolist(),
            "small_steps": self.small_steps,
            "large_steps": self.large_steps,
            "committed_steps": self.committed_steps,
            "epsilon_upper": epsilon,
            "best_order": best_order,
        }
