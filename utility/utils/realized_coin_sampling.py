"""Independent public branch coins and private Poisson sampling coins."""

from dataclasses import asdict, dataclass
import hashlib
import random
import secrets
from typing import Dict, Optional

import torch


def seed_commitment(seed: int) -> str:
    return hashlib.sha256(str(int(seed)).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class CoinDraw:
    draw_index: int
    uniform_value: float
    branch: str
    sigma: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RealizedCoinSampler:

    def __init__(
        self,
        p_large: float,
        sigma_small: float,
        sigma_large: float,
        seed: int,
    ):
        if not 0.0 <= p_large <= 1.0:
            raise ValueError("p_large must be in [0, 1].")
        if sigma_small <= 0.0 or sigma_large <= 0.0:
            raise ValueError("Both noise multipliers must be positive.")
        self.p_large = float(p_large)
        self.sigma_small = float(sigma_small)
        self.sigma_large = float(sigma_large)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.draw_count = 0

    def draw(self) -> CoinDraw:
        value = self._rng.random()
        branch = "large" if value < self.p_large else "small"
        sigma = self.sigma_large if branch == "large" else self.sigma_small
        self.draw_count += 1
        return CoinDraw(
            draw_index=self.draw_count,
            uniform_value=float(value),
            branch=branch,
            sigma=float(sigma),
        )


class PoissonStepSampler:

    def __init__(
        self,
        dataset_size: int,
        expected_batch_size: int,
        private_seed: Optional[int] = None,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if not 0 < expected_batch_size <= dataset_size:
            raise ValueError(
                "expected_batch_size must be in [1, dataset_size]."
            )
        if private_seed is None:
            private_seed = secrets.randbits(63)
        self.dataset_size = int(dataset_size)
        self.expected_batch_size = int(expected_batch_size)
        self.sample_rate = self.expected_batch_size / self.dataset_size
        self._private_seed = int(private_seed)
        self.private_seed_commitment = seed_commitment(self._private_seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self._private_seed)
        self.draw_count = 0

    def draw_indices(self) -> torch.Tensor:
        uniforms = torch.rand(
            self.dataset_size,
            generator=self._generator,
            device="cpu",
        )
        self.draw_count += 1
        return torch.nonzero(
            uniforms < self.sample_rate,
            as_tuple=False,
        ).flatten()

    def public_state(self) -> Dict[str, object]:
        return {
            "dataset_size": self.dataset_size,
            "expected_batch_size": self.expected_batch_size,
            "sample_rate": self.sample_rate,
            "draw_count": self.draw_count,
            "private_seed_commitment": self.private_seed_commitment,
        }
