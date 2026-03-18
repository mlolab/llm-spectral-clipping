import math
import torch
from typing import Optional, Callable, Literal

from .post_process import clip_sigvals, normalize_sigvals


class SPECTRA(torch.optim.Optimizer):
    """
    SPECTRA Wrapper that applies spectral post-processing to standard optimizer's updates.

    Uses the weight difference to recover the update direction U_k:
        U_k = (X_k - X_{k+1}) / η_k - λ * X_k

    Then recomputes: X_{k+1} = (1 - λη_k) * X_k - alpha * η_k * post_process(U_k)

    This works for decoupled weight decay formulations.

    Args:
        base_optimizer: The underlying optimizer (e.g., AdamW, Signum, AdEMAMix)
        post_process: Post-processing mode ("clip" or "normalize")
        clip_c: Clipping threshold for clip mode (default: 1.0)
        ns_steps: Number of Newton-Schulz iterations (default: 10)
        apply_to: Which parameters to apply post-processing to ("2d" or "all")
        warmup_steps: Number of warmup steps (default: 0, no warmup scheduling)
        base_lr: Base learning rate used to compute prod_lr_c = clip_c * base_lr
        disable_dynamic_clip: If True, always use constant clip_c (ignore warmup/decay schedules)
        clip_decay_type: Decay type for clipping threshold during decay phase
            ("constant", "linear", "sqrt", "cosine", "exp", "square")
        clip_decay_fract: Fraction of total_steps for the decay phase (default: 0.1)
        clip_final_scale: Final scale for c (c decays from clip_c to clip_c * clip_final_scale)
        total_steps: Total training iterations (needed for decay phase computation)

    Clipping threshold schedule (for WSD-style schedules):
        - Warmup phase (step < warmup_steps): c = prod_lr_c / current_lr (so c * lr = constant)
        - Stable phase (warmup_steps <= step < total_steps - decay_steps): c = clip_c (fixed)
        - Decay phase (last clip_decay_fract * total_steps steps): c decays from clip_c to clip_c * clip_final_scale
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        post_process: Literal["clip", "normalize"] = "clip",
        clip_c: float = 1.0,
        ns_steps: int = 10,
        apply_to: Literal["2d", "all"] = "2d",
        warmup_steps: int = 0,
        base_lr: float = 1e-3,
        disable_dynamic_clip: bool = False,
        clip_decay_type: str = "constant",
        clip_decay_fract: float = 0.1,
        clip_final_scale: float = 0.0,
        total_steps: int = 0,
    ):
        # Store base optimizer - we delegate param_groups and state to it
        self.base_optimizer = base_optimizer
        self.post_process_mode = post_process
        self.clip_c = clip_c
        self.ns_steps = ns_steps
        self.apply_to = apply_to

        # Warmup schedule for clipping threshold
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.prod_lr_c = clip_c * base_lr  # constant product during warmup
        self._step_count = 0
        self.disable_dynamic_clip = disable_dynamic_clip

        # Decay schedule for clipping threshold (for WSD-style schedules)
        self.clip_decay_type = clip_decay_type
        self.clip_decay_fract = clip_decay_fract
        self.clip_final_scale = clip_final_scale
        self.total_steps = total_steps

        # Initialize Optimizer base class with empty params/defaults
        # We override param_groups and state properties to delegate to base_optimizer
        # This is needed so that isinstance(self, Optimizer) returns True for schedulers
        self.defaults = base_optimizer.defaults
        self._param_groups = base_optimizer.param_groups
        self._state = base_optimizer.state

        # Build mapping from param to (group_idx, param_idx) for efficient lookup
        self._param_to_group = {}
        for group_idx, group in enumerate(base_optimizer.param_groups):
            for param_idx, p in enumerate(group["params"]):
                self._param_to_group[p] = (group_idx, param_idx)

    def _should_process(self, param: torch.Tensor) -> bool:
        """Determine if a parameter should have post-processing applied."""
        if self.apply_to == "all":
            return True
        elif self.apply_to == "2d":
            return param.ndim == 2
        return False

    def _get_param_config(self, param: torch.Tensor) -> tuple:
        """Get learning rate and weight decay for a parameter."""
        group_idx, _ = self._param_to_group[param]
        group = self.base_optimizer.param_groups[group_idx]
        lr = group.get("lr", 0.0)
        weight_decay = group.get("weight_decay", 0.0)
        return lr, weight_decay

    def _get_current_clip_c(self, current_lr: float) -> float:
        """
        Get the current clipping threshold based on step count and learning rate.

        For WSD-style schedules with three phases:
        - Warmup phase (step < warmup_steps): c = prod_lr_c / current_lr (so c * lr = constant)
        - Stable phase (warmup_steps <= step < n_hold): c = clip_c (fixed)
        - Decay phase (n_hold <= step < total_steps): c decays from clip_c to clip_c * clip_final_scale

        If disable_dynamic_clip is True, always return clip_c.
        If clip_decay_type is "constant", skip the decay phase (use constant clip_c after warmup).
        """
        if self.disable_dynamic_clip:
            return self.clip_c

        # Phase 1: Warmup - keep c * lr constant
        if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
            return self.prod_lr_c / max(current_lr, 1e-10)

        # Phase 2 & 3: After warmup
        if self.clip_decay_type == "constant":
            return self.clip_c

        # Compute decay phase boundaries
        n_anneal_steps = int(self.clip_decay_fract * self.total_steps)
        n_hold = self.total_steps - n_anneal_steps

        # Phase 2: Stable (hold) - constant c
        if self._step_count < n_hold:
            return self.clip_c

        # Phase 3: Decay - decay c from clip_c to clip_c * clip_final_scale
        progress = (self._step_count - n_hold) / max(n_anneal_steps, 1)
        progress = min(progress, 1.0)  # clamp to [0, 1]

        # Compute decay factor (1 -> clip_final_scale)
        final = self.clip_final_scale
        if self.clip_decay_type == "linear":
            factor = final + (1 - final) * (1 - progress)
        elif self.clip_decay_type == "sqrt":
            factor = final + (1 - final) * (1 - math.sqrt(progress))
        elif self.clip_decay_type == "cosine":
            factor = final + (1 - final) * (1 + math.cos(math.pi * progress)) * 0.5
        elif self.clip_decay_type == "exp":
            # Exponential decay: factor = final^progress if final > 0, else linear fallback
            factor = (final ** progress) if final > 0 else (1 - progress)
        elif self.clip_decay_type == "square":
            factor = final + (1 - final) * (1 - progress ** 2)
        else:
            # Fallback to constant
            factor = 1.0

        return self.clip_c * factor

    def _apply_post_process(self, U: torch.Tensor, current_lr: float) -> torch.Tensor:
        """Apply the selected post-processing to update U."""
        if self.post_process_mode == "clip":
            clip_c = self._get_current_clip_c(current_lr)
            return clip_sigvals(U, clip_c=clip_c, ns_iter=self.ns_steps)
        elif self.post_process_mode == "normalize":
            return normalize_sigvals(U, steps=self.ns_steps)
        else:
            raise ValueError(f"Unknown post_process mode: {self.post_process_mode}")

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step with spectral post-processing.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        # Identify target parameters and store X_k
        params_to_process = []
        X_k_copies = {}

        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and self._should_process(p):
                    params_to_process.append(p)
                    X_k_copies[p] = p.data.clone()

        # Call base optimizer step
        loss = self.base_optimizer.step(closure)

        # For each target param, recover U_k, apply post-processing, recompute X_{k+1}
        for p in params_to_process:
            X_k = X_k_copies[p]
            X_k1 = p.data  # Current value after base optimizer step
            lr, weight_decay = self._get_param_config(p)

            if lr == 0:
                continue

            # Recover U_k: U_k = (X_k - X_{k+1}) / lr - weight_decay * X_k
            U_k = (X_k - X_k1) / lr - weight_decay * X_k

            # Apply post-processing (pass lr for dynamic clip_c during warmup)
            U_k_processed = self._apply_post_process(U_k, lr)

            if U_k_processed.dim() >= 2:
                # Scale for rectangular matrices (like Muon)
                alpha = max(1, U_k_processed.size(-2) / U_k_processed.size(-1))**0.5
            else:
                alpha = 1.0

            # Recompute X_{k+1} = (1 - weight_decay * lr) * X_k - lr * α * U_k_processed
            p.data.copy_((1 - weight_decay * lr) * X_k - lr * alpha * U_k_processed)

        # Free X_k copies and increment step count
        del X_k_copies
        self._step_count += 1

        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set parameter groups (delegates to base optimizer)."""
        self._param_groups = value
        self.base_optimizer.param_groups = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        """Set state (delegates to base optimizer)."""
        self._state = value
        self.base_optimizer.state = value

    def state_dict(self):
        """Return state dict including wrapper state and base optimizer state."""
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        """Load state dict into wrapper and base optimizer."""
        # Handle both old format (direct base optimizer state) and new format (nested)
        if "base_optimizer" in state_dict:
            self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
            self._step_count = state_dict.get("_step_count", 0)
        else:
            # Backwards compatibility: old format was just base optimizer state
            self.base_optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        """Add a param group to the base optimizer."""
        self.base_optimizer.add_param_group(param_group)
        # Update param to group mapping
        group_idx = len(self.base_optimizer.param_groups) - 1
        for param_idx, p in enumerate(param_group["params"]):
            self._param_to_group[p] = (group_idx, param_idx)
