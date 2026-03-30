"""
SVD Recorder for tracking singular values of gradients and optimizer updates.

This module provides utilities for recording and analyzing the spectral structure
of gradients and updates during training.

Also includes NoiseStructureRecorder for analyzing the relationship between
"true" gradients (approximated via large batch accumulation) and stochastic
gradient noise.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from data.utils import DataReader


class SVDRecorder:
    """Records singular values of gradients and optimizer updates at specified steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize the SVD recorder.

        Args:
            model: The model being trained
            cfg: Configuration object with record_svd, svd_record_steps, svd_layers, iterations
            save_dir: Directory to save SVD recordings
        """
        self.cfg = cfg
        self.save_dir = Path(save_dir) if save_dir else None
        self.enabled = getattr(cfg, "record_svd", False)

        if not self.enabled:
            return

        # Compute which iteration numbers to record
        self.record_steps: Set[int] = set()
        svd_fractions = getattr(cfg, "svd_record_steps", [0.0, 0.05, 0.5, 0.99])
        for frac in svd_fractions:
            step = max(1, int(frac * cfg.iterations))
            self.record_steps.add(step)

        # Determine which layers to record based on model structure
        # layer_names: friendly_name -> full_param_name
        self.layer_names: Dict[str, str] = {}
        # reverse mapping: full_param_name -> friendly_name
        self.param_to_layer: Dict[str, str] = {}
        self._setup_layer_names(model, cfg)

        # Storage for recorded singular values
        # Structure: {step: {layer_name: {"grad": tensor, "update": tensor}}}
        self.records: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[SVDRecorder] Enabled. Will record at steps: {sorted(self.record_steps)}"
        )
        print(f"[SVDRecorder] Recording layers: {list(self.layer_names.keys())}")

    def _setup_layer_names(self, model, cfg):
        """Setup layer names to record based on configuration."""
        svd_layers = getattr(
            cfg, "svd_layers", ["embedding", "early", "middle", "late"]
        )
        n_layer = getattr(cfg, "n_layer", 12)

        # Map position names to layer indices
        layer_indices = {
            "early": 0,
            "middle": n_layer // 2,
            "late": n_layer - 1,
        }

        # Build mapping from friendly names to actual parameter names
        for pos in svd_layers:
            if pos == "embedding":
                self.layer_names["embedding"] = "transformer.wte.weight"
                self.param_to_layer["transformer.wte.weight"] = "embedding"
            elif pos in layer_indices:
                idx = layer_indices[pos]
                # Record attention and MLP layers
                attn_name = f"transformer.h.{idx}.attn.c_attn.weight"
                mlp_name = f"transformer.h.{idx}.mlp.w1.weight"
                self.layer_names[f"{pos}_attn"] = attn_name
                self.layer_names[f"{pos}_mlp"] = mlp_name
                self.param_to_layer[attn_name] = f"{pos}_attn"
                self.param_to_layer[mlp_name] = f"{pos}_mlp"

    def should_record(self, step: int) -> bool:
        """Check if SVD should be recorded at this step."""
        return self.enabled and step in self.record_steps

    def get_target_params(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.nn.Parameter]:
        """Get the target parameters to record."""
        if not self.enabled:
            return {}

        target_params = {}
        param_dict = dict(model.named_parameters())

        for friendly_name, param_name in self.layer_names.items():
            if param_name in param_dict:
                target_params[friendly_name] = param_dict[param_name]
            else:
                print(
                    f"[SVDRecorder] Warning: Parameter {param_name} not found in model"
                )

        return target_params

    @torch.no_grad()
    def compute_singular_values(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute singular values of a tensor.

        Args:
            tensor: 2D tensor (matrix) to compute SVD for

        Returns:
            1D tensor of singular values in descending order
        """
        if tensor.ndim == 1:
            # For 1D tensors (biases, norms), return the absolute values sorted
            return tensor.abs().sort(descending=True).values.cpu().float()

        # For 2D tensors, compute SVD
        # Use float32 for numerical stability
        tensor_float = tensor.float()

        # torch.linalg.svdvals computes only singular values (more efficient)
        try:
            singular_values = torch.linalg.svdvals(tensor_float)
        except RuntimeError:
            # Fallback for very large matrices - use randomized SVD approximation
            # For now, just compute full SVD
            _, singular_values, _ = torch.linalg.svd(tensor_float, full_matrices=False)

        return singular_values.cpu()

    @torch.no_grad()
    def record_gradients(self, model: torch.nn.Module, step: int):
        """
        Record singular values of gradients for target layers.

        Args:
            model: The model (gradients should already be computed via backward())
            step: Current training step
        """
        if not self.should_record(step):
            return

        if step not in self.records:
            self.records[step] = {}

        target_params = self.get_target_params(model)

        for name, param in target_params.items():
            if param.grad is None:
                print(f"[SVDRecorder] Warning: No gradient for {name} at step {step}")
                continue

            if name not in self.records[step]:
                self.records[step][name] = {}

            grad = param.grad.data
            sv = self.compute_singular_values(grad)
            self.records[step][name]["grad_sv"] = sv

            print(
                f"[SVDRecorder] Step {step}, {name} grad: "
                f"shape={tuple(grad.shape)}, "
                f"sv_max={sv[0].item():.4e}, sv_min={sv[-1].item():.4e}, "
                f"sv_ratio={sv[0].item()/sv[-1].item():.2e}"
            )

    @torch.no_grad()
    def record_update(self, name: str, update: torch.Tensor, step: int):
        """
        Record singular values of an optimizer update.

        Args:
            name: Friendly name of the layer
            update: The update tensor (before applying to weights)
            step: Current training step
        """
        if not self.should_record(step):
            return

        if name not in self.layer_names:
            return

        if step not in self.records:
            self.records[step] = {}

        if name not in self.records[step]:
            self.records[step][name] = {}

        sv = self.compute_singular_values(update)
        self.records[step][name]["update_sv"] = sv

        print(
            f"[SVDRecorder] Step {step}, {name} update: "
            f"shape={tuple(update.shape)}, "
            f"sv_max={sv[0].item():.4e}, sv_min={sv[-1].item():.4e}, "
            f"sv_ratio={sv[0].item()/sv[-1].item():.2e}"
        )

    @torch.no_grad()
    def store_weights_before_step(self, model: torch.nn.Module, step: int):
        """
        Store a copy of target parameters before optimizer step.

        This allows computing U_k = (X_k - X_{k+1})/η_k - λX_k after the step,
        which works for ANY optimizer with decoupled weight decay.

        Args:
            model: The model (before optimizer.step())
            step: Current training step
        """
        if not self.should_record(step):
            return

        # Store copies of target parameters
        self._stored_weights: Dict[str, torch.Tensor] = {}
        target_params = self.get_target_params(model)

        for name, param in target_params.items():
            self._stored_weights[name] = param.data.clone()

    @torch.no_grad()
    def compute_and_record_updates(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int
    ):
        """
        Compute optimizer updates from weight differences and record their SVD.

        For optimizers with decoupled weight decay:
            X_{k+1} = X_k - η_k * (λX_k + U_k)

        Rearranging to solve for U_k:
            U_k = (X_k - X_{k+1})/η_k - λX_k

        where:
        - X_k was stored before step via store_weights_before_step()
        - X_{k+1} is the current parameter value after step
        - η_k is the learning rate
        - λ is the weight decay coefficient

        Args:
            model: The model (after optimizer.step())
            optimizer: The optimizer (to get learning rate and weight decay)
            step: Current training step
        """
        if not self.should_record(step):
            return

        if not hasattr(self, "_stored_weights") or not self._stored_weights:
            print(f"[SVDRecorder] Warning: No stored weights for step {step}")
            return

        target_params = self.get_target_params(model)

        # Get learning rate and weight decay from optimizer (use first param group as default)
        lr = optimizer.param_groups[0]["lr"]
        weight_decay = optimizer.param_groups[0].get("weight_decay", 0.0)

        for name, param in target_params.items():
            if name not in self._stored_weights:
                continue

            X_k = self._stored_weights[name]
            X_k1 = param.data

            # U_k = (X_k - X_{k+1})/η_k - λX_k
            # This recovers the optimizer update direction (e.g., sign(momentum) for Signum,
            # m/(sqrt(v)+eps) for Adam) excluding the weight decay term
            update = (X_k - X_k1) / lr - weight_decay * X_k

            # Record the update SVD
            self.record_update(name, update, step)

        # Free stored weights
        self._stored_weights.clear()

    def save_records(self, step: int):
        """Save recorded singular values for a given step to disk."""
        if not self.enabled or self.save_dir is None:
            return

        if step not in self.records:
            return

        save_path = self.save_dir / f"svd_step_{step}.pt"
        torch.save(self.records[step], save_path)
        print(f"[SVDRecorder] Saved SVD records for step {step} to {save_path}")

    def save_all_records(self):
        """Save all recorded singular values to disk."""
        if not self.enabled or self.save_dir is None:
            return

        save_path = self.save_dir / "svd_all_records.pt"
        torch.save(self.records, save_path)
        print(f"[SVDRecorder] Saved all SVD records to {save_path}")

    def get_layer_name_for_param(self, param_name: str) -> Optional[str]:
        """Get the friendly layer name for a parameter name."""
        if param_name is None:
            return None
        return self.param_to_layer.get(param_name, None)


def create_svd_recorder(
    model: torch.nn.Module, cfg, exp_dir: Path
) -> Optional[SVDRecorder]:
    """
    Factory function to create an SVD recorder if enabled.

    Args:
        model: The model being trained
        cfg: Configuration object
        exp_dir: Experiment directory

    Returns:
        SVDRecorder instance if enabled, None otherwise
    """
    if not getattr(cfg, "record_svd", False):
        return None

    save_dir = cfg.svd_save_dir if cfg.svd_save_dir else exp_dir / "svd_records"
    return SVDRecorder(model, cfg, save_dir=save_dir)


class NoiseStructureRecorder:
    """
    Records noise structure analysis at specified training steps.

    This class analyzes the relationship between the "true" gradient G
    (approximated by averaging over many batches) and stochastic gradient noise.

    At each recording step:
    1. Compute G by accumulating gradients over many batches
    2. Store G's singular values and top-k singular vectors
    3. Sample stochastic gradients g, compute noise N = g - G
    4. Store N's singular values and alignment with G's top-k subspace
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg,
        data_src,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize the noise structure recorder.

        Args:
            model: The model being trained
            cfg: Configuration object with noise analysis params
            data_src: Data source to create an independent DataReader
            save_dir: Directory to save noise structure recordings
        """
        self.cfg = cfg
        self.save_dir = Path(save_dir) if save_dir else None
        self.enabled = getattr(cfg, "record_noise_structure", False)

        if not self.enabled:
            return

        # Compute which iteration numbers to record (reuse SVD recording steps)
        self.record_steps: Set[int] = set()
        svd_fractions = getattr(cfg, "svd_record_steps", [0.0, 0.05, 0.5, 0.99])
        for frac in svd_fractions:
            step = max(1, int(frac * cfg.iterations))
            self.record_steps.add(step)

        # Noise analysis parameters
        self.num_samples = getattr(cfg, "noise_num_samples", 4096)
        self.top_k = getattr(cfg, "noise_top_k", 5)
        self.num_repeats = getattr(cfg, "noise_num_repeats", 20)
        self.stochastic_batch_size = getattr(cfg, "noise_batch_size", 1)
        noise_seed = getattr(cfg, "noise_data_seed", 9999)

        # Create DataReader for true gradient G (uses training batch size for efficiency)
        self.true_grad_reader = DataReader(
            data_src=data_src,
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            seed=noise_seed,
            with_replacement=True,  # Important: sample with replacement
            auto_shard=False,  # Don't shard - master process only
            keep_in_ram=getattr(cfg, "data_in_ram", False),
        )

        # Create DataReader for stochastic gradient g (uses specified noise_batch_size)
        self.stochastic_reader = DataReader(
            data_src=data_src,
            batch_size=self.stochastic_batch_size,
            sequence_length=cfg.sequence_length,
            seed=noise_seed + 1,  # Different seed for stochastic samples
            with_replacement=True,
            auto_shard=False,
            keep_in_ram=getattr(cfg, "data_in_ram", False),
        )

        # Setup layer names (reuse logic from SVDRecorder)
        self.layer_names: Dict[str, str] = {}
        self.param_to_layer: Dict[str, str] = {}
        self._setup_layer_names(model, cfg)

        # Storage for recorded data
        self.records: Dict[int, Dict] = {}

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[NoiseStructureRecorder] Enabled. Will record at steps: {sorted(self.record_steps)}"
        )
        print(
            f"[NoiseStructureRecorder] Recording layers: {list(self.layer_names.keys())}"
        )
        print(
            f"[NoiseStructureRecorder] num_samples={self.num_samples}, top_k={self.top_k}, num_repeats={self.num_repeats}, stochastic_batch_size={self.stochastic_batch_size}"
        )

    def _setup_layer_names(self, model, cfg):
        """Setup layer names to record based on configuration (same logic as SVDRecorder)."""
        svd_layers = getattr(
            cfg, "svd_layers", ["embedding", "early", "middle", "late"]
        )
        n_layer = getattr(cfg, "n_layer", 12)

        layer_indices = {
            "early": 0,
            "middle": n_layer // 2,
            "late": n_layer - 1,
        }

        for pos in svd_layers:
            if pos == "embedding":
                self.layer_names["embedding"] = "transformer.wte.weight"
                self.param_to_layer["transformer.wte.weight"] = "embedding"
            elif pos in layer_indices:
                idx = layer_indices[pos]
                attn_name = f"transformer.h.{idx}.attn.c_attn.weight"
                mlp_name = f"transformer.h.{idx}.mlp.w1.weight"
                self.layer_names[f"{pos}_attn"] = attn_name
                self.layer_names[f"{pos}_mlp"] = mlp_name
                self.param_to_layer[attn_name] = f"{pos}_attn"
                self.param_to_layer[mlp_name] = f"{pos}_mlp"

    def should_record(self, step: int) -> bool:
        """Check if noise structure should be recorded at this step."""
        return self.enabled and step in self.record_steps

    def get_target_params(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.nn.Parameter]:
        """Get the target parameters to record."""
        if not self.enabled:
            return {}

        target_params = {}
        param_dict = dict(model.named_parameters())

        for friendly_name, param_name in self.layer_names.items():
            if param_name in param_dict:
                target_params[friendly_name] = param_dict[param_name]
            else:
                print(
                    f"[NoiseStructureRecorder] Warning: Parameter {param_name} not found in model"
                )

        return target_params

    @staticmethod
    def _get_batch(
        reader: DataReader, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from the data reader and move to device."""
        x, y = reader.sample_batch()
        return x.to(device), y.to(device)

    @torch.no_grad()
    def compute_full_svd_with_top_k(
        self,
        tensor: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full SVD and return singular values plus top-k singular vectors.

        Args:
            tensor: 2D tensor (matrix) to compute SVD for
            k: Number of top singular vectors to return

        Returns:
            Tuple of (singular_values, U_k, V_k) where:
            - singular_values: all singular values, shape (min(m,n),)
            - U_k: top-k left singular vectors, shape (m, k)
            - V_k: top-k right singular vectors, shape (n, k)
        """
        tensor_float = tensor.float()

        # Full SVD: U @ diag(S) @ V^H = tensor
        U, S, Vh = torch.linalg.svd(tensor_float, full_matrices=False)

        # Limit k to actual number of singular values
        actual_k = min(k, S.shape[0])

        # Return singular values and top-k vectors
        U_k = U[:, :actual_k].cpu()  # shape (m, actual_k)
        V_k = Vh[
            :actual_k, :
        ].T.cpu()  # shape (n, actual_k) - transpose because svd returns V^H
        singular_values = S.cpu()

        return singular_values, U_k, V_k

    @torch.no_grad()
    def compute_topk_svd(
        self,
        tensor: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-k singular values and vectors using randomized SVD.

        This is more efficient than full SVD when we only need top-k components.

        Args:
            tensor: 2D tensor (matrix) to compute SVD for
            k: Number of top singular values/vectors to compute

        Returns:
            Tuple of (top_k_singular_values, U_k, V_k) where:
            - top_k_singular_values: top-k singular values, shape (k,)
            - U_k: top-k left singular vectors, shape (m, k)
            - V_k: top-k right singular vectors, shape (n, k)
        """
        tensor_float = tensor.float()

        # Use randomized SVD for efficiency
        U, S, V = torch.svd_lowrank(tensor_float, q=k)

        return S.cpu(), U.cpu(), V.cpu()

    @torch.no_grad()
    def compute_subspace_distance(
        self,
        U1: torch.Tensor,  # shape (m, k)
        U2: torch.Tensor,  # shape (m, k)
    ) -> Tuple[float, float]:
        """
        Compute subspace distances between two orthonormal matrices.

        Returns two metrics:
        1. Spectral distance: sqrt(λ_max(B)) = max|sin(θ_i)| — worst-case misalignment
        2. Chordal distance: sqrt(trace(B)/k) = RMS of sin(θ_i) — average misalignment

        Where B = I - A @ A^T, A = U1^T @ U2, and θ_i are the principal angles.

        Both metrics are in [0, 1]:
        - 0: subspaces are identical
        - 1: subspaces are orthogonal

        Efficient computation (O(k³) instead of O(m²)):
        - A = U1^T @ U2  (k x k)
        - B = I - A @ A^T  (k x k)
        - spectral_dist = sqrt(max_eigenvalue(B))
        - chordal_dist = sqrt(trace(B) / k)

        Args:
            U1: First orthonormal matrix, shape (m, k)
            U2: Second orthonormal matrix, shape (m, k)

        Returns:
            Tuple of (spectral_distance, chordal_distance), both in [0, 1]
        """
        k = U1.shape[1]
        A = U1.T @ U2  # shape (k, k)
        B = torch.eye(k, device=A.device, dtype=A.dtype) - A @ A.T  # shape (k, k)

        # Spectral distance: sqrt(max eigenvalue of B)
        # B is symmetric PSD, so eigenvalues = singular values
        eigenvalues = torch.linalg.eigvalsh(B.float())
        max_eigenvalue = eigenvalues[-1].item()  # eigvalsh returns in ascending order
        max_eigenvalue = max(
            0.0, min(1.0, max_eigenvalue)
        )  # clamp for numerical stability
        spectral_dist = math.sqrt(max_eigenvalue)

        # Chordal distance: sqrt(trace(B) / k)
        trace_B = torch.trace(B).item()
        trace_B = max(0.0, min(float(k), trace_B))  # clamp for numerical stability
        chordal_dist = math.sqrt(trace_B / k)

        return spectral_dist, chordal_dist

    def compute_true_gradient(
        self,
        model: torch.nn.Module,
        type_ctx,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the "true" gradient G by averaging over many batches.

        This accumulates gradients over num_samples samples without updating weights.

        Args:
            model: The model (weights are not modified)
            type_ctx: Type context for mixed precision
            device: Device to use for computation

        Returns:
            Dict mapping layer names to accumulated gradient tensors
        """
        target_params = self.get_target_params(model)

        # Initialize accumulators
        grad_accum = {
            name: torch.zeros_like(param.data) for name, param in target_params.items()
        }

        num_batches = self.num_samples // self.cfg.batch_size

        # Reset true grad reader step to ensure fresh samples
        self.true_grad_reader.set_step(0)

        for batch_idx in range(num_batches):
            # Zero gradients
            model.zero_grad(set_to_none=True)

            # Sample batch from true grad reader
            x, y = self._get_batch(self.true_grad_reader, device)

            # Forward and backward pass
            with type_ctx:
                outputs = model(x, targets=y, moe=getattr(self.cfg, "moe", False))
            loss = outputs["loss"]
            loss.backward()

            # Accumulate gradients for target layers
            for name, param in target_params.items():
                if param.grad is not None:
                    grad_accum[name] += param.grad.data

        # Average gradients
        for name in grad_accum:
            grad_accum[name] /= num_batches

        # Clean up
        model.zero_grad(set_to_none=True)

        return grad_accum

    def compute_stochastic_gradient(
        self,
        model: torch.nn.Module,
        type_ctx,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute a single stochastic gradient g using the configured stochastic batch size.

        Args:
            model: The model
            type_ctx: Type context for mixed precision
            device: Device to use

        Returns:
            Dict mapping layer names to gradient tensors
        """
        target_params = self.get_target_params(model)

        model.zero_grad(set_to_none=True)

        # Sample a fresh batch from stochastic reader (uses noise_batch_size)
        x, y = self._get_batch(self.stochastic_reader, device)

        # Forward and backward
        with type_ctx:
            outputs = model(x, targets=y, moe=getattr(self.cfg, "moe", False))
        loss = outputs["loss"]
        loss.backward()

        # Extract gradients
        grads = {}
        for name, param in target_params.items():
            if param.grad is not None:
                grads[name] = param.grad.data.clone()

        model.zero_grad(set_to_none=True)

        return grads

    def record_noise_structure(
        self,
        model: torch.nn.Module,
        step: int,
        type_ctx,
        device: str,
    ):
        """
        Perform noise structure analysis at the current training step.

        This method:
        1. Computes the "true" gradient G by averaging over many samples
        2. Computes full SVD of G and stores top-k singular vectors (U_G, V_G)
        3. Repeatedly samples stochastic gradients g (single batch each), computes noise N = g - G
        4. For each N, computes top-k SVD to get (U_N, V_N) and top-k singular values
        5. Computes subspace distance: max{dist(U_N, U_G), dist(V_N, V_G)}
           where dist(U, V) = ||UU^T - VV^T||_2 (principal angle distance, in [0, 1])
        6. Tracks the maximum noise spectral norm seen across all samples

        Args:
            model: The model (weights are frozen during analysis)
            step: Current training step
            type_ctx: Type context for mixed precision
            device: Device to use for computation
        """
        if not self.should_record(step):
            return

        print(
            f"\n[NoiseStructureRecorder] Step {step}: Starting noise structure analysis..."
        )

        # Save current training state
        was_training = model.training
        model.eval()

        # Step 1: Compute true gradient G
        num_batches = self.num_samples // self.cfg.batch_size
        print(
            f"[NoiseStructureRecorder] Computing true gradient G over {self.num_samples} samples ({num_batches} batches)..."
        )
        true_grads = self.compute_true_gradient(model, type_ctx, device)

        # Initialize record for this step
        step_record = {
            "step": step,
            "layers": {},
            "metadata": {
                "num_samples_for_G": self.num_samples,
                "batch_size_for_G": self.cfg.batch_size,
                "stochastic_batch_size": self.stochastic_batch_size,
                "top_k": self.top_k,
                "num_repeats": self.num_repeats,
                "noise_data_seed": getattr(self.cfg, "noise_data_seed", 9999),
            },
        }

        # Step 2 & 3: For each layer, compute SVD of G and analyze noise
        for layer_name, G in true_grads.items():
            print(
                f"[NoiseStructureRecorder] Processing layer: {layer_name} (shape={tuple(G.shape)})"
            )

            # Compute full SVD of G to get top-k singular vectors
            sv_G, U_k, V_k = self.compute_full_svd_with_top_k(G, self.top_k)

            layer_record = {
                "true_grad": {
                    "singular_values": sv_G,
                    "top_k_U": U_k,
                    "top_k_V": V_k,
                    "spectral_norm": sv_G[0].item(),
                },
                "noise_samples": [],
                "max_noise_spectral_norm": 0.0,  # Track max ||N||_2 seen
            }

            print(
                f"  [G] ||G||_2 = {sv_G[0].item():.4e}, condition = {sv_G[0].item()/sv_G[-1].item():.2e}"
            )

            # Step 3: Sample noise and analyze subspace distance (repeat many times)
            max_noise_sv = 0.0
            for repeat_idx in range(self.num_repeats):
                # Get stochastic gradient (single batch)
                stochastic_grads = self.compute_stochastic_gradient(
                    model, type_ctx, device
                )

                if layer_name not in stochastic_grads:
                    continue

                g = stochastic_grads[layer_name]

                # Compute noise N = g - G
                N = g - G

                # Compute top-k SVD of noise
                sv_N, U_N, V_N = self.compute_topk_svd(N, self.top_k)

                # Track max noise spectral norm
                top_sv_N = sv_N[0].item()
                if top_sv_N > max_noise_sv:
                    max_noise_sv = top_sv_N

                # Compute subspace distances between N's and G's principal subspaces
                # Returns (spectral_dist, chordal_dist) for each side
                spec_dist_left, chord_dist_left = self.compute_subspace_distance(
                    U_N, U_k
                )
                spec_dist_right, chord_dist_right = self.compute_subspace_distance(
                    V_N, V_k
                )
                spectral_dist = max(spec_dist_left, spec_dist_right)
                chordal_dist = max(chord_dist_left, chord_dist_right)

                noise_sample = {
                    "top_k_singular_values": sv_N,  # shape (k,)
                    "spectral_distance": spectral_dist,  # max|sin(θ_i)| — worst-case
                    "chordal_distance": chordal_dist,  # RMS of sin(θ_i) — average
                }
                layer_record["noise_samples"].append(noise_sample)

                if (repeat_idx + 1) % 5 == 0 or repeat_idx == 0:
                    print(
                        f"  [Repeat {repeat_idx+1}/{self.num_repeats}] ||N||_2 = {top_sv_N:.4e}, spectral={spectral_dist:.4f}, chordal={chordal_dist:.4f}"
                    )

            layer_record["max_noise_spectral_norm"] = max_noise_sv
            step_record["layers"][layer_name] = layer_record

            # Print summary for this layer
            all_top_sv = [
                ns["top_k_singular_values"][0].item()
                for ns in layer_record["noise_samples"]
            ]
            all_spectral = [
                ns["spectral_distance"] for ns in layer_record["noise_samples"]
            ]
            all_chordal = [
                ns["chordal_distance"] for ns in layer_record["noise_samples"]
            ]
            print(
                f"  [Summary] max ||N||_2 = {max_noise_sv:.4e} (vs ||G||_2 = {sv_G[0].item():.4e}, ratio = {max_noise_sv/sv_G[0].item():.2f})"
            )
            print(
                f"  [Summary] ||N||_2: mean={np.mean(all_top_sv):.4e}, std={np.std(all_top_sv):.4e}"
            )
            print(
                f"  [Summary] spectral_dist: mean={np.mean(all_spectral):.4f}, std={np.std(all_spectral):.4f}"
            )
            print(
                f"  [Summary] chordal_dist: mean={np.mean(all_chordal):.4f}, std={np.std(all_chordal):.4f}"
            )

        # Restore model state
        if was_training:
            model.train()

        # Store record
        self.records[step] = step_record

        # Save to disk
        self.save_records(step)

        print(
            f"[NoiseStructureRecorder] Step {step}: Noise structure analysis complete.\n"
        )

    def save_records(self, step: int):
        """Save noise structure records for a given step to disk."""
        if not self.enabled or self.save_dir is None:
            return

        if step not in self.records:
            return

        save_path = self.save_dir / f"noise_step_{step}.pt"
        torch.save(self.records[step], save_path)
        print(
            f"[NoiseStructureRecorder] Saved noise records for step {step} to {save_path}"
        )

    def save_all_records(self):
        """Save all recorded noise structure data to disk."""
        if not self.enabled or self.save_dir is None:
            return

        if not self.records:
            return

        save_path = self.save_dir / "noise_all_records.pt"
        torch.save(self.records, save_path)
        print(f"[NoiseStructureRecorder] Saved all noise records to {save_path}")


def create_noise_recorder(
    model: torch.nn.Module,
    cfg,
    data_src,
    exp_dir: Path,
) -> Optional[NoiseStructureRecorder]:
    """
    Factory function to create a noise structure recorder if enabled.

    Args:
        model: The model being trained
        cfg: Configuration object
        data_src: Training data source (np.memmap, np.ndarray, or path)
        exp_dir: Experiment directory

    Returns:
        NoiseStructureRecorder instance if enabled, None otherwise
    """
    if not getattr(cfg, "record_noise_structure", False):
        return None

    save_dir = exp_dir / "noise_records"
    return NoiseStructureRecorder(model, cfg, data_src, save_dir=save_dir)


class UpdateNoiseRecorder:
    """
    Records noise structure analysis for optimizer updates at specified training steps.

    This class analyzes the relationship between the "true" optimizer update U
    (computed using large-batch gradient) and stochastic update noise.

    At each recording step:
    1. Save current weights and optimizer state
    2. Compute "true" update U by:
       - Accumulating gradients over many batches
       - Running optimizer.step()
       - Recovering U = (X_k - X_{k+1})/lr - wd*X_k
    3. For each noise sample:
       - Restore weights and optimizer state
       - Compute stochastic gradient (small batch)
       - Run optimizer.step()
       - Recover u = (X_k - X_{k+1})/lr - wd*X_k
       - Compute noise N = u - U
       - Record N's singular values and subspace distance from U
    4. Restore weights and optimizer state to original
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg,
        data_src,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize the update noise recorder.

        Args:
            model: The model being trained
            optimizer: The optimizer being used
            cfg: Configuration object with noise analysis params
            data_src: Data source to create an independent DataReader
            save_dir: Directory to save update noise recordings
        """
        self.cfg = cfg
        self.save_dir = Path(save_dir) if save_dir else None
        self.enabled = getattr(cfg, "record_update_noise", False)
        self.optimizer = optimizer

        if not self.enabled:
            return

        # Compute which iteration numbers to record (reuse SVD recording steps)
        self.record_steps: Set[int] = set()
        svd_fractions = getattr(cfg, "svd_record_steps", [0.0, 0.05, 0.5, 0.99])
        for frac in svd_fractions:
            step = max(1, int(frac * cfg.iterations))
            self.record_steps.add(step)

        # Reuse existing noise analysis parameters
        self.num_samples = getattr(cfg, "noise_num_samples", 4096)
        self.top_k = getattr(cfg, "noise_top_k", 5)
        self.num_repeats = getattr(cfg, "noise_num_repeats", 20)
        self.stochastic_batch_size = getattr(cfg, "noise_batch_size", 1)
        noise_seed = getattr(cfg, "noise_data_seed", 9999)

        # Create DataReader for true update (uses training batch size for efficiency)
        self.true_grad_reader = DataReader(
            data_src=data_src,
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            seed=noise_seed + 100,  # Different seed from gradient noise recorder
            with_replacement=True,
            auto_shard=False,
            keep_in_ram=getattr(cfg, "data_in_ram", False),
        )

        # Create DataReader for stochastic updates (uses specified noise_batch_size)
        self.stochastic_reader = DataReader(
            data_src=data_src,
            batch_size=self.stochastic_batch_size,
            sequence_length=cfg.sequence_length,
            seed=noise_seed + 101,
            with_replacement=True,
            auto_shard=False,
            keep_in_ram=getattr(cfg, "data_in_ram", False),
        )

        # Setup layer names (reuse logic from NoiseStructureRecorder)
        self.layer_names: Dict[str, str] = {}
        self.param_to_layer: Dict[str, str] = {}
        self._setup_layer_names(model, cfg)

        # Storage for recorded data
        self.records: Dict[int, Dict] = {}

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[UpdateNoiseRecorder] Enabled. Will record at steps: {sorted(self.record_steps)}"
        )
        print(
            f"[UpdateNoiseRecorder] Recording layers: {list(self.layer_names.keys())}"
        )
        print(
            f"[UpdateNoiseRecorder] num_samples={self.num_samples}, top_k={self.top_k}, "
            f"num_repeats={self.num_repeats}, stochastic_batch_size={self.stochastic_batch_size}"
        )

    def _setup_layer_names(self, model, cfg):
        """Setup layer names to record based on configuration (same logic as NoiseStructureRecorder)."""
        svd_layers = getattr(
            cfg, "svd_layers", ["embedding", "early", "middle", "late"]
        )
        n_layer = getattr(cfg, "n_layer", 12)

        layer_indices = {
            "early": 0,
            "middle": n_layer // 2,
            "late": n_layer - 1,
        }

        for pos in svd_layers:
            if pos == "embedding":
                self.layer_names["embedding"] = "transformer.wte.weight"
                self.param_to_layer["transformer.wte.weight"] = "embedding"
            elif pos in layer_indices:
                idx = layer_indices[pos]
                attn_name = f"transformer.h.{idx}.attn.c_attn.weight"
                mlp_name = f"transformer.h.{idx}.mlp.w1.weight"
                self.layer_names[f"{pos}_attn"] = attn_name
                self.layer_names[f"{pos}_mlp"] = mlp_name
                self.param_to_layer[attn_name] = f"{pos}_attn"
                self.param_to_layer[mlp_name] = f"{pos}_mlp"

    def should_record(self, step: int) -> bool:
        """Check if update noise structure should be recorded at this step."""
        return self.enabled and step in self.record_steps

    def get_target_params(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.nn.Parameter]:
        """Get the target parameters to record."""
        if not self.enabled:
            return {}

        target_params = {}
        param_dict = dict(model.named_parameters())

        for friendly_name, param_name in self.layer_names.items():
            if param_name in param_dict:
                target_params[friendly_name] = param_dict[param_name]
            else:
                print(
                    f"[UpdateNoiseRecorder] Warning: Parameter {param_name} not found in model"
                )

        return target_params

    @staticmethod
    def _get_batch(
        reader: DataReader, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from the data reader and move to device."""
        x, y = reader.sample_batch()
        return x.to(device), y.to(device)

    def save_weights(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Clone all model parameters."""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    def restore_weights(
        self, model: torch.nn.Module, saved_weights: Dict[str, torch.Tensor]
    ):
        """Restore model parameters from saved dict."""
        restored_count = 0
        for name, param in model.named_parameters():
            if name in saved_weights:
                param.data.copy_(saved_weights[name])
                restored_count += 1
            else:
                print(
                    f"[UpdateNoiseRecorder] WARNING: Parameter {name} not found in saved_weights!"
                )

        # Sanity check
        total_params = sum(1 for _ in model.named_parameters())
        if restored_count != total_params:
            print(
                f"[UpdateNoiseRecorder] WARNING: Only restored {restored_count}/{total_params} parameters!"
            )

    def _get_base_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """Get the base optimizer if wrapped by SPECTRA."""
        if hasattr(optimizer, "base_optimizer"):
            return optimizer.base_optimizer
        return optimizer

    def create_temp_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """
        Create a temporary optimizer instance for recording analysis.

        This creates a fresh optimizer of the SAME TYPE as the real optimizer
        with the same hyperparameters and copies the momentum buffers. This avoids
        modifying the real training optimizer's internal state, which is critical
        for fused optimizers that have special CUDA kernel state not captured by
        state_dict().

        If the optimizer is wrapped with SPECTRA, we preserve that wrapper
        so that the recorded updates match what's actually used in training.

        Args:
            model: The model being trained
            optimizer: The real training optimizer

        Returns:
            A fresh optimizer of the same type with copied momentum state
        """
        # Check if optimizer is wrapped with SPECTRA BEFORE unwrapping
        is_spectral = (
            hasattr(optimizer, "base_optimizer")
            and type(optimizer).__name__ == "SPECTRA"
        )
        spectral_config = None

        if is_spectral:
            # Save spectral configuration to re-apply after creating base temp optimizer
            spectral_config = {
                "post_process": optimizer.post_process_mode,
                "clip_c": optimizer.clip_c,
                "ns_steps": optimizer.ns_steps,
                "apply_to": optimizer.apply_to,
                "warmup_steps": optimizer.warmup_steps,
                "base_lr": optimizer.base_lr,
                "_step_count": optimizer._step_count,
            }

        base_opt = self._get_base_optimizer(optimizer)
        opt_class = type(base_opt)
        opt_class_name = opt_class.__name__

        # Build param groups with same hyperparams
        param_groups = []
        for group in base_opt.param_groups:
            new_group = {k: v for k, v in group.items() if k != "params"}
            new_group["params"] = list(group["params"])
            param_groups.append(new_group)

        # Create fresh optimizer of the same type
        if opt_class_name == "AdamW" or opt_class_name == "Adam":
            temp_opt = torch.optim.AdamW(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.999)),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                fused=False,  # Disable fused for temp optimizer
            )
        elif opt_class_name == "Signum":
            from .sign import Signum

            temp_opt = Signum(
                param_groups,
                lr=base_opt.defaults["lr"],
                momentum=base_opt.defaults.get("momentum", 0),
                dampening=base_opt.defaults.get("dampening", 0),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                nesterov=base_opt.defaults.get("nesterov", False),
                sign_update=base_opt.defaults.get("sign_update", True),
            )
        elif opt_class_name == "SGD":
            temp_opt = torch.optim.SGD(
                param_groups,
                lr=base_opt.defaults["lr"],
                momentum=base_opt.defaults.get("momentum", 0),
                dampening=base_opt.defaults.get("dampening", 0),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                nesterov=base_opt.defaults.get("nesterov", False),
            )
        elif opt_class_name == "AdEMAMix":
            from .ademamix import AdEMAMix

            temp_opt = AdEMAMix(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.999, 0.9999)),
                alpha=base_opt.defaults.get("alpha", 5.0),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
            )
        elif opt_class_name == "Lion":
            from .lion import Lion

            temp_opt = Lion(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.99)),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
            )
        elif opt_class_name == "ADOPT":
            from .adopt import ADOPT

            temp_opt = ADOPT(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.9999)),
                eps=base_opt.defaults.get("eps", 1e-6),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                decouple=base_opt.defaults.get("decouple", True),
            )
        elif opt_class_name == "MARS":
            from .mars import MARS

            temp_opt = MARS(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.95, 0.99)),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                gamma=base_opt.defaults.get("gamma", 0.025),
                is_approx=base_opt.defaults.get("is_approx", True),
                mars_type=base_opt.defaults.get("mars_type", "mars-adamw"),
            )
        elif opt_class_name == "Lamb":
            from .lamb import Lamb

            temp_opt = Lamb(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.999)),
                eps=base_opt.defaults.get("eps", 1e-6),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                bias_correction=base_opt.defaults.get("bias_correction", False),
            )
        elif opt_class_name == "SophiaG":
            from .sophia import SophiaG

            temp_opt = SophiaG(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.965, 0.99)),
                rho=base_opt.defaults.get("rho", 0.04),
                weight_decay=base_opt.defaults.get("weight_decay", 0.1),
            )
        elif opt_class_name == "SOAP":
            from .soap import SOAP

            temp_opt = SOAP(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.95, 0.95)),
                shampoo_beta=base_opt.defaults.get("shampoo_beta", -1),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.01),
                precondition_frequency=base_opt.defaults.get(
                    "precondition_frequency", 10
                ),
                max_precond_dim=base_opt.defaults.get("max_precond_dim", 10000),
                merge_dims=base_opt.defaults.get("merge_dims", False),
                precondition_1d=base_opt.defaults.get("precondition_1d", False),
                normalize_grads=base_opt.defaults.get("normalize_grads", False),
                correct_bias=base_opt.defaults.get("correct_bias", True),
            )
        elif opt_class_name == "Prodigy":
            from .prodigy import Prodigy

            temp_opt = Prodigy(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.999)),
                beta3=base_opt.defaults.get("beta3", None),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                decouple=base_opt.defaults.get("decouple", True),
                use_bias_correction=base_opt.defaults.get("use_bias_correction", False),
                safeguard_warmup=base_opt.defaults.get("safeguard_warmup", False),
            )
        elif opt_class_name == "Adafactor":
            from .adafactor import Adafactor

            temp_opt = Adafactor(
                param_groups,
                lr=base_opt.defaults.get("lr", 1e-3),
                eps2=base_opt.defaults.get("eps2", (1e-30, 1e-3)),
                clip_threshold=base_opt.defaults.get("clip_threshold", 1.0),
                decay_rate=base_opt.defaults.get("decay_rate", -0.8),
                beta1=base_opt.defaults.get("beta1", None),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                scale_parameter=base_opt.defaults.get("scale_parameter", True),
                relative_step=base_opt.defaults.get("relative_step", True),
                warmup_init=base_opt.defaults.get("warmup_init", False),
            )
        elif opt_class_name == "AdamWScheduleFree":
            from .schedulefree import AdamWScheduleFree

            temp_opt = AdamWScheduleFree(
                param_groups,
                lr=base_opt.defaults["lr"],
                betas=base_opt.defaults.get("betas", (0.9, 0.999)),
                eps=base_opt.defaults.get("eps", 1e-8),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                warmup_steps=base_opt.defaults.get("warmup_steps", 0),
                r=base_opt.defaults.get("r", 0.0),
                weight_lr_power=base_opt.defaults.get("weight_lr_power", 2.0),
            )
        elif opt_class_name == "SGDScheduleFree":
            from .schedulefree import SGDScheduleFree

            temp_opt = SGDScheduleFree(
                param_groups,
                lr=base_opt.defaults["lr"],
                momentum=base_opt.defaults.get("momentum", 0.9),
                weight_decay=base_opt.defaults.get("weight_decay", 0.0),
                warmup_steps=base_opt.defaults.get("warmup_steps", 0),
                r=base_opt.defaults.get("r", 0.0),
                weight_lr_power=base_opt.defaults.get("weight_lr_power", 2.0),
            )
        elif opt_class_name in ("Muon", "DMuon"):
            # Muon has complex internal structure, use generic instantiation
            print(
                f"[UpdateNoiseRecorder] Warning: '{opt_class_name}' has complex structure, "
                f"attempting generic instantiation"
            )
            temp_opt = opt_class(param_groups, **base_opt.defaults)
        elif opt_class_name in ("Scion", "MomentumScion"):
            # Scion also has complex structure with norm types
            print(
                f"[UpdateNoiseRecorder] Warning: '{opt_class_name}' has complex structure, "
                f"attempting generic instantiation"
            )
            temp_opt = opt_class(param_groups, **base_opt.defaults)
        else:
            # Fallback: try to instantiate with defaults
            # This may not work for all optimizers but covers common cases
            print(
                f"[UpdateNoiseRecorder] Warning: Unknown optimizer type '{opt_class_name}', "
                f"attempting generic instantiation"
            )
            temp_opt = opt_class(param_groups, **base_opt.defaults)

        # Copy momentum buffers from real optimizer to temp optimizer
        for param in model.parameters():
            if param in base_opt.state:
                real_state = base_opt.state[param]
                temp_state = {}
                for key, val in real_state.items():
                    if isinstance(val, torch.Tensor):
                        temp_state[key] = val.clone()
                    else:
                        temp_state[key] = val
                temp_opt.state[param] = temp_state

        # If the original optimizer was wrapped with SPECTRA, wrap the temp optimizer too
        if is_spectral and spectral_config is not None:
            from .spectra import SPECTRA

            temp_opt = SPECTRA(
                temp_opt,
                post_process=spectral_config["post_process"],
                clip_c=spectral_config["clip_c"],
                ns_steps=spectral_config["ns_steps"],
                apply_to=spectral_config["apply_to"],
                warmup_steps=spectral_config["warmup_steps"],
                base_lr=spectral_config["base_lr"],
            )
            # Copy the step count so the clipping threshold schedule is in sync
            temp_opt._step_count = spectral_config["_step_count"]

        return temp_opt

    @torch.no_grad()
    def compute_full_svd_with_top_k(
        self,
        tensor: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full SVD and return singular values plus top-k singular vectors.

        Args:
            tensor: 2D tensor (matrix) to compute SVD for
            k: Number of top singular vectors to return

        Returns:
            Tuple of (singular_values, U_k, V_k)
        """
        tensor_float = tensor.float()
        U, S, Vh = torch.linalg.svd(tensor_float, full_matrices=False)
        actual_k = min(k, S.shape[0])
        U_k = U[:, :actual_k].cpu()
        V_k = Vh[:actual_k, :].T.cpu()
        singular_values = S.cpu()
        return singular_values, U_k, V_k

    @torch.no_grad()
    def compute_topk_svd(
        self,
        tensor: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-k singular values and vectors using randomized SVD.
        """
        tensor_float = tensor.float()
        U, S, V = torch.svd_lowrank(tensor_float, q=k)
        return S.cpu(), U.cpu(), V.cpu()

    @torch.no_grad()
    def compute_subspace_distance(
        self,
        U1: torch.Tensor,
        U2: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute subspace distances between two orthonormal matrices.

        Returns:
            Tuple of (spectral_distance, chordal_distance), both in [0, 1]
        """
        k = U1.shape[1]
        A = U1.T @ U2
        B = torch.eye(k, device=A.device, dtype=A.dtype) - A @ A.T

        eigenvalues = torch.linalg.eigvalsh(B.float())
        max_eigenvalue = eigenvalues[-1].item()
        max_eigenvalue = max(0.0, min(1.0, max_eigenvalue))
        spectral_dist = math.sqrt(max_eigenvalue)

        trace_B = torch.trace(B).item()
        trace_B = max(0.0, min(float(k), trace_B))
        chordal_dist = math.sqrt(trace_B / k)

        return spectral_dist, chordal_dist

    def compute_update(
        self,
        model: torch.nn.Module,
        temp_optimizer: torch.optim.Optimizer,
        type_ctx,
        device: str,
        data_reader: DataReader,
        num_batches: int,
        saved_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute optimizer update given a gradient computed from data_reader.

        This method uses a temporary optimizer that won't affect the real training
        optimizer. The model weights are restored to saved_weights before computation.

        Steps:
        1. Restore weights to saved values
        2. Accumulate gradients over num_batches batches
        3. Average gradients (divide by num_batches)
        4. Call temp_optimizer.step()
        5. Recover U = (X_k - X_{k+1})/lr - wd*X_k for target layers
        6. Return U dict

        Args:
            model: The model
            temp_optimizer: A temporary optimizer (not the real training optimizer)
            type_ctx: Type context for mixed precision
            device: Device to use
            data_reader: DataReader to sample batches from
            num_batches: Number of batches to accumulate
            saved_weights: Saved weights to restore before computation

        Returns:
            Dict mapping layer names to update tensors
        """
        # Step 1: Restore weights to saved values
        self.restore_weights(model, saved_weights)
        model.zero_grad(set_to_none=True)

        # Step 2: Accumulate gradients over num_batches batches
        for batch_idx in range(num_batches):
            x, y = self._get_batch(data_reader, device)
            with type_ctx:
                outputs = model(x, targets=y, moe=getattr(self.cfg, "moe", False))
            loss = outputs["loss"]
            loss.backward()

        # Step 3: Average gradients
        if num_batches > 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= num_batches

        # Step 4: Get lr and wd, then call optimizer.step()
        lr = temp_optimizer.param_groups[0]["lr"]
        weight_decay = temp_optimizer.param_groups[0].get("weight_decay", 0.0)

        # Store X_k for target layers before step
        target_params = self.get_target_params(model)
        X_k = {
            name: saved_weights[self.layer_names[name]].clone()
            for name in target_params.keys()
        }

        temp_optimizer.step()

        # Step 5: Recover U = (X_k - X_{k+1})/lr - wd*X_k
        updates = {}
        for name, param in target_params.items():
            X_k_layer = X_k[name]
            X_k1 = param.data
            # U = (X_k - X_{k+1})/lr - wd*X_k
            U = (X_k_layer - X_k1) / lr - weight_decay * X_k_layer
            updates[name] = U

        return updates

    def record_update_noise_structure(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        type_ctx,
        device: str,
    ):
        """
        Perform update noise structure analysis at the current training step.

        This method uses a TEMPORARY optimizer for all analysis, keeping the real
        training optimizer completely untouched. This is easier to be implemented for
        fused optimizers (like fused AdamW) that have internal CUDA kernel state not
        captured bystate_dict(), which would be corrupted by save/restore operations.

        1. Save current model weights
        2. Create a temporary optimizer with copied momentum state
        3. Compute "true" update U using large-batch gradient + temp optimizer step
        4. For each noise sample, compute stochastic update u, then N = u - U
        5. Record N's singular values and subspace alignment with U
        6. Restore model weights

        Args:
            model: The model (weights will be temporarily modified but restored)
            optimizer: The real training optimizer (NOT modified - only used to copy state)
            step: Current training step
            type_ctx: Type context for mixed precision
            device: Device to use for computation
        """
        if not self.should_record(step):
            return

        print(
            f"\n[UpdateNoiseRecorder] Step {step}: Starting update noise structure analysis..."
        )

        # Save current training state
        was_training = model.training
        model.eval()

        # STEP 0: Save model weights (NOT optimizer state - we use temp optimizer)
        print(f"[UpdateNoiseRecorder] Saving weights...")
        saved_weights = self.save_weights(model)

        # Create a temporary optimizer with copied momentum state
        # This avoids modifying the real training optimizer entirely
        print(f"[UpdateNoiseRecorder] Creating temporary optimizer...")
        temp_optimizer = self.create_temp_optimizer(model, optimizer)

        # Get optimizer info for metadata
        base_opt = self._get_base_optimizer(optimizer)
        lr = base_opt.param_groups[0]["lr"]
        weight_decay = base_opt.param_groups[0].get("weight_decay", 0.0)
        opt_name = getattr(self.cfg, "opt", "unknown")

        step_record = None
        try:
            # STEP 1: Compute true update U
            num_batches = self.num_samples // self.cfg.batch_size
            print(
                f"[UpdateNoiseRecorder] Computing true update U over {self.num_samples} samples ({num_batches} batches)..."
            )
            self.true_grad_reader.set_step(0)

            true_updates = self.compute_update(
                model,
                temp_optimizer,
                type_ctx,
                device,
                self.true_grad_reader,
                num_batches,
                saved_weights,
            )

            # Initialize record for this step
            step_record = {
                "step": step,
                "layers": {},
                "metadata": {
                    "optimizer": opt_name,
                    "num_samples_for_U": self.num_samples,
                    "batch_size_for_U": self.cfg.batch_size,
                    "stochastic_batch_size": self.stochastic_batch_size,
                    "top_k": self.top_k,
                    "num_repeats": self.num_repeats,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                },
            }

            # STEP 2: For each layer, compute SVD of U and analyze noise
            for layer_name, U in true_updates.items():
                print(
                    f"[UpdateNoiseRecorder] Processing layer: {layer_name} (shape={tuple(U.shape)})"
                )

                # Compute full SVD of U to get top-k singular vectors
                sv_U, U_k, V_k = self.compute_full_svd_with_top_k(U, self.top_k)

                layer_record = {
                    "true_update": {
                        "singular_values": sv_U,
                        "top_k_U": U_k,
                        "top_k_V": V_k,
                        "spectral_norm": sv_U[0].item(),
                    },
                    "noise_samples": [],
                    "max_noise_spectral_norm": 0.0,
                }

                print(
                    f"  [U] ||U||_2 = {sv_U[0].item():.4e}, condition = {sv_U[0].item()/sv_U[-1].item():.2e}"
                )

                # STEP 3: Sample stochastic updates and analyze noise
                max_noise_sv = 0.0
                for repeat_idx in range(self.num_repeats):
                    # IMPORTANT: Create a fresh temp optimizer for each noise sample
                    # This is necessary because optimizer.step() modifies the momentum state,
                    # and we need each stochastic update to start from the same initial state
                    temp_optimizer_stochastic = self.create_temp_optimizer(
                        model, optimizer
                    )

                    # Compute stochastic update (single small batch)
                    stochastic_updates = self.compute_update(
                        model,
                        temp_optimizer_stochastic,
                        type_ctx,
                        device,
                        self.stochastic_reader,
                        1,  # Single batch
                        saved_weights,
                    )

                    # Discard temp optimizer for this sample
                    del temp_optimizer_stochastic

                    if layer_name not in stochastic_updates:
                        continue

                    u = stochastic_updates[layer_name]

                    # Compute noise N = u - U
                    N = u - U

                    # Compute top-k SVD of noise
                    sv_N, U_N, V_N = self.compute_topk_svd(N, self.top_k)

                    # Track max noise spectral norm
                    top_sv_N = sv_N[0].item()
                    if top_sv_N > max_noise_sv:
                        max_noise_sv = top_sv_N

                    # Compute subspace distances between N's and U's principal subspaces
                    spec_dist_left, chord_dist_left = self.compute_subspace_distance(
                        U_N, U_k
                    )
                    spec_dist_right, chord_dist_right = self.compute_subspace_distance(
                        V_N, V_k
                    )
                    spectral_dist = max(spec_dist_left, spec_dist_right)
                    chordal_dist = max(chord_dist_left, chord_dist_right)

                    noise_sample = {
                        "top_k_singular_values": sv_N,
                        "spectral_distance": spectral_dist,
                        "chordal_distance": chordal_dist,
                    }
                    layer_record["noise_samples"].append(noise_sample)

                    if (repeat_idx + 1) % 5 == 0 or repeat_idx == 0:
                        print(
                            f"  [Repeat {repeat_idx+1}/{self.num_repeats}] ||N||_2 = {top_sv_N:.4e}, "
                            f"spectral={spectral_dist:.4f}, chordal={chordal_dist:.4f}"
                        )

                layer_record["max_noise_spectral_norm"] = max_noise_sv
                step_record["layers"][layer_name] = layer_record

                # Print summary for this layer
                all_top_sv = [
                    ns["top_k_singular_values"][0].item()
                    for ns in layer_record["noise_samples"]
                ]
                all_spectral = [
                    ns["spectral_distance"] for ns in layer_record["noise_samples"]
                ]
                all_chordal = [
                    ns["chordal_distance"] for ns in layer_record["noise_samples"]
                ]
                print(
                    f"  [Summary] max ||N||_2 = {max_noise_sv:.4e} (vs ||U||_2 = {sv_U[0].item():.4e}, "
                    f"ratio = {max_noise_sv/sv_U[0].item():.2f})"
                )
                print(
                    f"  [Summary] ||N||_2: mean={np.mean(all_top_sv):.4e}, std={np.std(all_top_sv):.4e}"
                )
                print(
                    f"  [Summary] spectral_dist: mean={np.mean(all_spectral):.4f}, std={np.std(all_spectral):.4f}"
                )
                print(
                    f"  [Summary] chordal_dist: mean={np.mean(all_chordal):.4f}, std={np.std(all_chordal):.4f}"
                )

        finally:
            # STEP 4: Final cleanup - restore weights only (optimizer was never touched!)
            print(f"[UpdateNoiseRecorder] Restoring weights...")
            self.restore_weights(model, saved_weights)

            # Verify weights were restored correctly
            max_diff = 0.0
            for name, param in model.named_parameters():
                if name in saved_weights:
                    diff = (param.data - saved_weights[name]).abs().max().item()
                    if diff > max_diff:
                        max_diff = diff
            print(f"[DEBUG] After restore: weights max_diff = {max_diff:.6e}")

            model.zero_grad(set_to_none=True)

            # Discard the temporary optimizer (it goes out of scope automatically)
            del temp_optimizer

            if was_training:
                model.train()

        # Store record (only if we successfully completed recording)
        if step_record is not None:
            self.records[step] = step_record
            # Save to disk
            self.save_records(step)

        print(
            f"[UpdateNoiseRecorder] Step {step}: Update noise structure analysis complete.\n"
        )

    def save_records(self, step: int):
        """Save update noise records for a given step to disk."""
        if not self.enabled or self.save_dir is None:
            return

        if step not in self.records:
            return

        save_path = self.save_dir / f"update_noise_step_{step}.pt"
        torch.save(self.records[step], save_path)
        print(
            f"[UpdateNoiseRecorder] Saved update noise records for step {step} to {save_path}"
        )

    def save_all_records(self):
        """Save all recorded update noise data to disk."""
        if not self.enabled or self.save_dir is None:
            return

        if not self.records:
            return

        save_path = self.save_dir / "update_noise_all_records.pt"
        torch.save(self.records, save_path)
        print(f"[UpdateNoiseRecorder] Saved all update noise records to {save_path}")


def create_update_noise_recorder(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg,
    data_src,
    exp_dir: Path,
) -> Optional[UpdateNoiseRecorder]:
    """
    Factory function to create an update noise recorder if enabled.

    Args:
        model: The model being trained
        optimizer: The optimizer being used
        cfg: Configuration object
        data_src: Training data source (np.memmap, np.ndarray, or path)
        exp_dir: Experiment directory

    Returns:
        UpdateNoiseRecorder instance if enabled, None otherwise
    """
    if not getattr(cfg, "record_update_noise", False):
        return None

    save_dir = exp_dir / "update_noise_records"
    return UpdateNoiseRecorder(model, optimizer, cfg, data_src, save_dir=save_dir)
