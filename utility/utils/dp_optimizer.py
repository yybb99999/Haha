
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop



def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(
            self,
            l2_norm_clip,
            noise_multiplier,
            minibatch_size,
            microbatch_size,
            noise_mode='gaussian',
            sigma_large=15.0,
            p_large=0.05,
            large_step_update='sgd_bypass_scaled',
            large_step_lr_scale=0.1,
            *args,
            **kwargs
        ):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier  
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            self.noise_mode = noise_mode
            self.sigma_large = float(sigma_large)
            self.p_large = float(p_large)
            self.large_step_update = large_step_update
            self.large_step_lr_scale = float(large_step_lr_scale)

            
            if self.noise_mode not in ['gaussian', 'elementwise_gmm', 'stepwise_gmm']:
                raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

            if self.large_step_update not in ['normal', 'sgd_bypass', 'sgd_bypass_scaled', 'skip']:
                raise ValueError(
                    f"Unknown large_step_update: {self.large_step_update}. "
                    f"Expected one of ['normal', 'sgd_bypass', 'sgd_bypass_scaled', 'skip']."
                )

            if self.large_step_lr_scale < 0.0:
                raise ValueError(
                    f"large_step_lr_scale must be non-negative, got {self.large_step_lr_scale}"
                )

            if self.noise_mode in ['elementwise_gmm', 'stepwise_gmm']:
                if self.sigma_large < self.noise_multiplier:
                    raise ValueError(
                        f"sigma_large should be >= sigma_small/noise_multiplier. "
                        f"Got sigma_large={self.sigma_large}, "
                        f"sigma_small={self.noise_multiplier}."
                    )
                if not (0.0 <= self.p_large <= 1.0):
                    raise ValueError(f"p_large must be in [0, 1], got {self.p_large}")

            
            self.dp_update_steps = 0

            self.stepwise_total_steps = 0
            self.stepwise_large_steps = 0
            self.stepwise_small_steps = 0
            self._last_step_sigma = None
            self._last_step_is_large = False

            for id, group in enumerate(self.param_groups):
                group['accum_grads'] = [
                    torch.zeros_like(param.data) if param.requires_grad else None
                    for param in group['params']
                ]
        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()


        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def _sample_step_sigma(self):

            if self.noise_mode != 'stepwise_gmm':
                self._last_step_sigma = float(self.noise_multiplier)
                self._last_step_is_large = False
                return float(self.noise_multiplier)

            use_large = torch.rand(1).item() < self.p_large

            self.stepwise_total_steps += 1

            if use_large:
                self.stepwise_large_steps += 1
                self._last_step_is_large = True
                self._last_step_sigma = float(self.sigma_large)
            else:
                self.stepwise_small_steps += 1
                self._last_step_is_large = False
                self._last_step_sigma = float(self.noise_multiplier)

            return self._last_step_sigma

        def _make_noise(self, grad_tensor, step_sigma=None):
            if self.noise_mode == 'gaussian':
                return (
                    self.l2_norm_clip
                    * float(self.noise_multiplier)
                    * torch.randn_like(grad_tensor)
                )

            elif self.noise_mode == 'elementwise_gmm':
                standard_gaussian = torch.randn_like(grad_tensor)
                rand_mask = torch.rand_like(grad_tensor)

                sigma_small_tensor = torch.full_like(
                    grad_tensor,
                    fill_value=float(self.noise_multiplier),
                )
                sigma_large_tensor = torch.full_like(
                    grad_tensor,
                    fill_value=float(self.sigma_large),
                )

                sigma_mask = torch.where(
                    rand_mask < self.p_large,
                    sigma_large_tensor,
                    sigma_small_tensor,
                )

                return self.l2_norm_clip * sigma_mask * standard_gaussian

            elif self.noise_mode == 'stepwise_gmm':
                if step_sigma is None:
                    raise ValueError(
                        "step_sigma must be provided when noise_mode='stepwise_gmm'."
                    )

                return (
                    self.l2_norm_clip
                    * float(step_sigma)
                    * torch.randn_like(grad_tensor)
                )

            else:
                raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        def _large_step_manual_update(self):
            if self.large_step_update == 'skip':
                return

            if self.large_step_update == 'sgd_bypass':
                lr_scale = 1.0
            elif self.large_step_update == 'sgd_bypass_scaled':
                lr_scale = float(self.large_step_lr_scale)
            else:
                raise ValueError(
                    f"_large_step_manual_update should not be called for "
                    f"large_step_update={self.large_step_update}"
                )

            with torch.no_grad():
                for group in self.param_groups:
                    lr = group.get('lr', 0.0)

                    for param in group['params']:
                        if param.requires_grad and param.grad is not None:
                            param.data.add_(
                                param.grad.data,
                                alpha=-lr * lr_scale,
                            )

        def step_dp(self, *args, **kwargs):
            step_sigma = self._sample_step_sigma()

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:

                        if param.grad is None:
                            param.grad = torch.zeros_like(param.data)
                        param.grad.data = accum_grad.clone()

                        noise = self._make_noise(
                            param.grad.data,
                            step_sigma=step_sigma,
                        )
                        param.grad.data.add_(noise)

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            if (
                self.noise_mode == 'stepwise_gmm'
                and self._last_step_is_large
                and self.large_step_update != 'normal'
            ):
                self._large_step_manual_update()
            else:
                super(DPOptimizerClass, self).step(*args, **kwargs)

            self.dp_update_steps += 1


        def step_dp_agd(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)

def get_dp_optimizer(
    dataset_name,
    algortithm,
    lr,
    momentum,
    C_t,
    sigma,
    batch_size,
    model,
    noise_mode='gaussian',
    sigma_large=15.0,
    p_large=0.05,
    large_step_update='sgd_bypass_scaled',
    large_step_lr_scale=0.1,
):

    if dataset_name == 'IMDB' and algortithm != 'DPAGD':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            noise_mode=noise_mode,
            sigma_large=sigma_large,
            p_large=p_large,
            large_step_update=large_step_update,
            large_step_lr_scale=large_step_lr_scale,
            params=model.parameters(),
            lr=lr,
        )
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            noise_mode=noise_mode,
            sigma_large=sigma_large,
            p_large=p_large,
            large_step_update=large_step_update,
            large_step_lr_scale=large_step_lr_scale,
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
    return optimizer
