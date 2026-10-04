"""DP-SGD optimizer that consumes a pre-authorized StepMix branch."""

from dataclasses import dataclass
import hashlib
import math
import secrets
from typing import Dict, Optional

import torch

from utils.dp_optimizer import DPSGD_Optimizer


@dataclass(frozen=True)
class StepAuthorization:
    token: str
    branch: str
    sigma: float


class PreselectedStepDPSGDOptimizer(DPSGD_Optimizer):

    def __init__(self, *args, noise_seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.noise_mode != "stepwise_gmm":
            raise ValueError(
                "PreselectedStepDPSGDOptimizer requires stepwise_gmm."
            )
        if noise_seed is None:
            noise_seed = secrets.randbits(63)
        self._noise_seed = int(noise_seed)
        self.noise_seed_commitment = hashlib.sha256(
            str(self._noise_seed).encode("ascii")
        ).hexdigest()
        self._noise_generators: Dict[str, torch.Generator] = {}
        self._pending_authorization: Optional[StepAuthorization] = None
        self._active_authorization: Optional[StepAuthorization] = None
        self._completed_token: Optional[str] = None

    def authorize_step(self, token: str, branch: str, sigma: float) -> None:
        if self._pending_authorization is not None:
            raise RuntimeError("An optimizer authorization is already pending.")
        if self._active_authorization is not None:
            raise RuntimeError("An optimizer authorization is active.")
        if self._completed_token is not None:
            raise RuntimeError("The previous optimizer step is not confirmed.")
        if branch not in {"small", "large"}:
            raise ValueError(f"Unknown branch: {branch}")
        expected = (
            float(self.sigma_large)
            if branch == "large"
            else float(self.noise_multiplier)
        )
        if not math.isclose(float(sigma), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Authorized sigma {sigma} does not match {branch} sigma "
                f"{expected}."
            )
        self._pending_authorization = StepAuthorization(
            token=str(token),
            branch=branch,
            sigma=expected,
        )

    def cancel_authorization(self, token: str) -> None:
        authorization = self._pending_authorization
        if authorization is None or authorization.token != token:
            raise RuntimeError("Optimizer authorization token mismatch.")
        self._pending_authorization = None

    def _sample_step_sigma(self) -> float:
        authorization = self._pending_authorization
        if authorization is None:
            raise RuntimeError(
                "A privacy-filter authorization is required before step_dp()."
            )
        self._pending_authorization = None
        self._active_authorization = authorization
        self.stepwise_total_steps += 1
        self._last_step_is_large = authorization.branch == "large"
        self._last_step_sigma = float(authorization.sigma)
        if self._last_step_is_large:
            self.stepwise_large_steps += 1
        else:
            self.stepwise_small_steps += 1
        return self._last_step_sigma

    def _generator_for(self, tensor: torch.Tensor) -> torch.Generator:
        device_key = str(tensor.device)
        generator = self._noise_generators.get(device_key)
        if generator is None:
            material = f"{self._noise_seed}:{device_key}".encode("ascii")
            digest = hashlib.sha256(material).digest()
            device_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(device_seed)
            self._noise_generators[device_key] = generator
        return generator

    def _make_noise(self, grad_tensor, step_sigma=None):
        if step_sigma is None:
            raise ValueError("A preselected step_sigma is required.")
        standard_gaussian = torch.randn(
            grad_tensor.shape,
            dtype=grad_tensor.dtype,
            device=grad_tensor.device,
            generator=self._generator_for(grad_tensor),
        )
        return self.l2_norm_clip * float(step_sigma) * standard_gaussian

    def step_dp(self, *args, **kwargs):
        if self._pending_authorization is None:
            raise RuntimeError(
                "step_dp() called without privacy-filter authorization."
            )
        token = self._pending_authorization.token
        updates_before = self.dp_update_steps
        try:
            super().step_dp(*args, **kwargs)
        except Exception:
            self._active_authorization = None
            raise
        if self.dp_update_steps != updates_before + 1:
            self._active_authorization = None
            raise RuntimeError("Authorized optimizer step did not commit once.")
        self._active_authorization = None
        self._completed_token = token

    def confirm_completed_step(self, token: str) -> None:
        if self._completed_token != token:
            raise RuntimeError("Completed optimizer token mismatch.")
        self._completed_token = None


def get_preselected_step_dpsgd_optimizer(
    lr,
    momentum,
    C_t,
    sigma_small,
    sigma_large,
    p_large,
    batch_size,
    model,
    large_step_update="sgd_bypass_scaled",
    large_step_lr_scale=0.1,
    noise_seed=None,
):
    return PreselectedStepDPSGDOptimizer(
        l2_norm_clip=C_t,
        noise_multiplier=sigma_small,
        minibatch_size=batch_size,
        microbatch_size=1,
        noise_mode="stepwise_gmm",
        sigma_large=sigma_large,
        p_large=p_large,
        large_step_update=large_step_update,
        large_step_lr_scale=large_step_lr_scale,
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        noise_seed=noise_seed,
    )
