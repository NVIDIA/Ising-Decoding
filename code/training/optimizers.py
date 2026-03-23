# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Optimizer and learning rate scheduler implementations.
"""
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                p.add_(update.sign_(), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class DebugLion(Optimizer):
    r"""Lion optimizer with NaN/Inf gradient detection and optional logging.
    Implements the official Lion algorithm: https://arxiv.org/abs/2302.06675
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, log_nan=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, log_nan=log_nan)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            log_nan = group.get("log_nan", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                # Check for NaNs/Infs
                if not torch.isfinite(grad).all():
                    if log_nan:
                        print("[DebugLion] Skipping update due to NaN or Inf in gradients.")
                    continue

                # Step weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # State init
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # Compute update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-lr)

                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def get_lr_scheduler(cfg, optimizer, total_steps):
    """Create a learning rate scheduler based on configuration.
    
    Args:
        cfg: Configuration object with lr_scheduler settings
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler instance
    """
    warmup_steps = cfg.lr_scheduler.warmup_steps

    if cfg.lr_scheduler.type == "warmup_then_decay":
        milestones_frac = cfg.lr_scheduler.milestones
        milestone_steps = [int(total_steps * frac) for frac in milestones_frac]
        gamma = cfg.lr_scheduler.gamma

        def linear_warmup_then_decay(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            decay_factor = 1.0
            for milestone_step in milestone_steps:
                if current_step >= milestone_step:
                    decay_factor *= gamma
            return decay_factor

        return LambdaLR(optimizer, lr_lambda=linear_warmup_then_decay)

    elif cfg.lr_scheduler.type == "cosine":
        min_lr_ratio = cfg.lr_scheduler.min_lr / cfg.optimizer.lr

        def cosine_with_warmup(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step -
                             warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

        return LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

    else:
        raise ValueError(f"Unknown lr_scheduler.type: {cfg.lr_scheduler.type}")
