from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch

from .quantizer import Quantizer
from ..transforms.transforms import BaseTransform, build_transform, get_transform_matrices

# We keep the default search space focused on practical transforms for FP4 group quantization.
DEFAULT_SEARCH_TRANSFORMS = ["identity", "hadamard", "dct", "dst", "gsr", "householder"]
SUPPORTED_SEARCH_TRANSFORMS = {"identity", "hadamard", "dct", "dst", "gsr", "householder", "fast_food"}
SUPPORTED_SEARCH_OBJECTIVES = {"auto", "mse", "cov", "jtail"}
SUPPORTED_SEARCH_BASE_LOSSES = {"mse", "cov"}
SUPPORTED_TAIL_WEIGHT_MODES = {"a_low", "b_high", "mixed_uniform", "mixed_middle", "two_tail", "auto_abm"}


class MixedGroupTransform(BaseTransform):
    """
    A transform that applies a potentially different matrix per quantization group.
    """

    def __init__(
        self,
        forward_matrices: torch.Tensor,
        backward_matrices: torch.Tensor,
        group_size: int,
        selected_transforms: Sequence[str],
        selected_tail_weight_modes: Sequence[str] | None = None,
    ):
        super().__init__()
        if forward_matrices.shape != backward_matrices.shape:
            raise ValueError("forward_matrices and backward_matrices must have identical shape.")
        if forward_matrices.ndim != 3:
            raise ValueError("Matrices must have shape [num_groups, group_size, group_size].")

        self.group_size = group_size
        self.num_groups = forward_matrices.shape[0]
        self.selected_transforms = list(selected_transforms)

        if selected_tail_weight_modes is None:
            self.selected_tail_weight_modes = []
        else:
            if len(selected_tail_weight_modes) != self.num_groups:
                raise ValueError(
                    "selected_tail_weight_modes must be None or have length equal to num_groups."
                )
            self.selected_tail_weight_modes = list(selected_tail_weight_modes)

        # Keep canonical matrices in fp32 buffers, then cast on-the-fly to input dtype/device.
        self.register_buffer("forward_matrices", forward_matrices.to(torch.float32), persistent=False)
        self.register_buffer("backward_matrices", backward_matrices.to(torch.float32), persistent=False)

        # New: strict linear-equivalence guard for each group block.
        # For row-vector convention used by F.linear, we need: x' = xT and W' = W T^{-T}.
        # This implies T @ (T^{-T})^T = I, i.e., forward @ backward^T should be identity.
        composed = self.forward_matrices.matmul(self.backward_matrices.transpose(-1, -2))
        eye = torch.eye(self.group_size, device=composed.device, dtype=composed.dtype).unsqueeze(0)
        max_equiv_error = (composed - eye).abs().max().item()
        if max_equiv_error > 1e-3:
            raise ValueError(
                f"MixedGroupTransform forward/backward mismatch; max |T@T_invT^T-I| = {max_equiv_error:.2e}"
            )

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim < 0:
            dim += x.ndim

        moved = False
        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)
            moved = True

        expected_last_dim = self.num_groups * self.group_size
        if x.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Last dim mismatch for MixedGroupTransform: got {x.shape[-1]}, expected {expected_last_dim}."
            )

        # Reshape into [batch_like, num_groups, group_size] and apply per-group matrices.
        x_view = x.reshape(-1, self.num_groups, self.group_size)
        matrices = self.backward_matrices if inv_t else self.forward_matrices
        out = torch.einsum("bgi,gij->bgj", x_view, matrices.to(device=x.device, dtype=x.dtype))
        out = out.reshape(*x.shape)

        if moved:
            out = out.movedim(-1, dim)

        return out

    def remove_parametrizations(self) -> None:
        pass

    def summary(self) -> dict[str, int]:
        return dict(Counter(self.selected_transforms))

    def tail_weight_mode_summary(self) -> dict[str, int]:
        if not self.selected_tail_weight_modes:
            return {}
        return dict(Counter(self.selected_tail_weight_modes))


def _validate_candidates(candidates: Sequence[str]) -> list[str]:
    ordered_unique = []
    seen = set()
    for name in candidates:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)

    invalid = [name for name in ordered_unique if name not in SUPPORTED_SEARCH_TRANSFORMS]
    if invalid:
        raise ValueError(
            f"Unsupported transform_search_candidates: {invalid}. "
            f"Supported: {sorted(SUPPORTED_SEARCH_TRANSFORMS)}"
        )

    return ordered_unique


def _compute_quantization_mse(x: torch.Tensor, quantizer_kwargs: dict[str, Any]) -> float:
    # Build a fresh quantizer for each score to avoid sharing internal scale-tracking state.
    quantizer = Quantizer(**quantizer_kwargs)
    scales, zeros = quantizer.get_quantization_params(x)
    x_q = quantizer(x, scales, zeros)
    return (x_q - x).pow(2).mean().item()


def _compute_quantization_delta(x: torch.Tensor, quantizer_kwargs: dict[str, Any]) -> torch.Tensor:
    quantizer = Quantizer(**quantizer_kwargs)
    scales, zeros = quantizer.get_quantization_params(x)
    x_q = quantizer(x, scales, zeros)
    return x_q - x


def _compute_gptq_consistent_group_error(
    rotated_group_weight: torch.Tensor,
    quantizer_kwargs: dict[str, Any],
    rotated_group_covariance: torch.Tensor,
    delta_w: torch.Tensor | None = None,
) -> float:
    """
    GPTQ-consistent local objective:
        Tr(DeltaW * Cov' * DeltaW^T) / out_features
    where:
        DeltaW = Q(W') - W'
        Cov'   = E[(xT)^T (xT)] for the current group.
    """
    if delta_w is None:
        # Fresh quantizer per score keeps global-scale/stat tracking isolated across candidates.
        delta_w = _compute_quantization_delta(rotated_group_weight, quantizer_kwargs)

    # Weighted output-error proxy (matches GPTQ's Hessian-weighted spirit).
    weighted_error = torch.einsum("oi,ij,oj->", delta_w, rotated_group_covariance, delta_w)
    # Normalize by out_features so layers with larger row count do not dominate shared-slot search.
    return (weighted_error / max(delta_w.shape[0], 1)).item()


def _build_tail_bin_weights(
    tail_bins: int,
    tail_weight_mode: str,
    tail_weight_power: float,
    device: torch.device,
) -> torch.Tensor:
    if tail_weight_mode not in SUPPORTED_TAIL_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported tail_weight_mode '{tail_weight_mode}'. "
            f"Supported: {sorted(SUPPORTED_TAIL_WEIGHT_MODES)}"
        )
    if tail_weight_power <= 0:
        raise ValueError(f"tail_weight_power must be > 0, got {tail_weight_power}.")
    if tail_weight_mode == "auto_abm":
        # auto_abm must be resolved to a concrete mode before building alpha.
        raise ValueError("tail_weight_mode auto_abm must be resolved per-group before weight building.")

    idx = torch.arange(tail_bins, device=device, dtype=torch.float32)
    if tail_weight_mode == "a_low":
        # A-type: emphasize low-quantile bins.
        alpha = ((tail_bins - idx) / max(tail_bins, 1)).pow(tail_weight_power)
    elif tail_weight_mode == "b_high":
        # B-type: emphasize high-quantile bins.
        alpha = ((idx + 1.0) / max(tail_bins, 1)).pow(tail_weight_power)
    elif tail_weight_mode == "mixed_uniform":
        # Mixed-type: uniform bin weights.
        alpha = torch.ones(tail_bins, device=device, dtype=torch.float32)
    elif tail_weight_mode == "mixed_middle":
        # Mixed-type: emphasize middle quantile bins.
        center = (tail_bins - 1.0) / 2.0
        denom = max(center, 1.0)
        dist = (idx - center).abs() / denom
        alpha = (1.0 - dist).clamp_min(0.0).pow(tail_weight_power) + 1e-3
    else:
        # Legacy mode: emphasize both tails.
        center = (tail_bins - 1.0) / 2.0
        denom = max(center, 1.0)
        dist = (idx - center).abs() / denom
        alpha = dist.clamp_min(0.0).pow(tail_weight_power) + 1e-3

    return alpha / alpha.clamp_min(1e-12).mean()


def _select_tail_weight_mode_for_group(
    group_weights: Sequence[torch.Tensor],
) -> str:
    """
    Auto-select A/B/mixed mode from group-block distribution.

    Heuristic (based on |W_g| quantiles):
    - A-type (a_low): sharp outliers above already-high values.
    - B-type (b_high): high values dominate most of the block.
    - Mixed: fallback.
    """
    if len(group_weights) == 0:
        return "mixed_uniform"

    merged = torch.cat([w.reshape(-1).abs().to(torch.float32) for w in group_weights], dim=0)
    if merged.numel() == 0:
        return "mixed_uniform"

    q50, q90, q99 = torch.quantile(merged, torch.tensor([0.50, 0.90, 0.99], device=merged.device)).tolist()
    eps = 1e-8
    outlier_sharpness = q99 / (q90 + eps)
    bulk_to_peak = q50 / (q99 + eps)
    high_to_peak = q90 / (q99 + eps)

    # A-type: few sharp outliers + low bulk level.
    if outlier_sharpness >= 1.6 and bulk_to_peak <= 0.30:
        return "a_low"
    # B-type: most values are already high.
    if bulk_to_peak >= 0.45 and high_to_peak >= 0.80:
        return "b_high"
    # Mixed fallback.
    return "mixed_uniform"


def _compute_tail_quantile_error(
    delta_w: torch.Tensor,
    reference_weight: torch.Tensor,
    tail_bins: int,
    tail_weight_mode: str,
    tail_weight_power: float,
) -> float:
    """
    Quantile tail term:
        L_tail = sum_b alpha_b * MSE_b
    where bins are formed on |reference_weight| quantiles.
    """
    if tail_bins <= 1:
        return delta_w.pow(2).mean().item()

    sq_err = delta_w.pow(2).reshape(-1).to(torch.float32)
    abs_ref = reference_weight.abs().reshape(-1).to(torch.float32)
    if sq_err.numel() == 0:
        return 0.0

    quantiles = torch.linspace(0.0, 1.0, tail_bins + 1, device=abs_ref.device, dtype=abs_ref.dtype)
    bin_edges = torch.quantile(abs_ref, quantiles)
    if not torch.isfinite(bin_edges).all():
        return sq_err.mean().item()

    # Internal edges produce bucket ids in [0, tail_bins-1].
    bucket_ids = torch.bucketize(abs_ref, bin_edges[1:-1], right=False)
    alpha = _build_tail_bin_weights(
        tail_bins=tail_bins,
        tail_weight_mode=tail_weight_mode,
        tail_weight_power=tail_weight_power,
        device=abs_ref.device,
    )

    weighted = torch.tensor(0.0, device=abs_ref.device, dtype=torch.float32)
    normalizer = torch.tensor(0.0, device=abs_ref.device, dtype=torch.float32)
    for bin_idx in range(tail_bins):
        bin_mask = bucket_ids == bin_idx
        if bin_mask.any():
            bin_mse = sq_err[bin_mask].mean()
        else:
            # Empty bin fallback keeps objective stable for degenerate distributions.
            bin_mse = sq_err.mean()
        weighted = weighted + alpha[bin_idx] * bin_mse
        normalizer = normalizer + alpha[bin_idx]

    return (weighted / normalizer.clamp_min(1e-12)).item()


def should_collect_group_covariances(
    objective: str,
    base_loss: str,
    auto_default: str,
) -> bool:
    resolved_objective = auto_default if objective == "auto" else objective
    if resolved_objective == "cov":
        return True
    if resolved_objective == "jtail" and base_loss == "cov":
        return True
    return False


def resolve_transform_search_objective(
    objective: str,
    base_loss: str,
    has_group_covariances: bool,
    auto_default: str,
) -> tuple[str, str]:
    if objective not in SUPPORTED_SEARCH_OBJECTIVES:
        raise ValueError(
            f"Unsupported transform_search objective '{objective}'. "
            f"Supported: {sorted(SUPPORTED_SEARCH_OBJECTIVES)}"
        )
    if base_loss not in SUPPORTED_SEARCH_BASE_LOSSES:
        raise ValueError(
            f"Unsupported transform_search base loss '{base_loss}'. "
            f"Supported: {sorted(SUPPORTED_SEARCH_BASE_LOSSES)}"
        )
    if auto_default not in {"mse", "cov"}:
        raise ValueError(f"auto_default must be 'mse' or 'cov', got '{auto_default}'.")

    resolved_objective = auto_default if objective == "auto" else objective
    resolved_base_loss = base_loss if resolved_objective == "jtail" else resolved_objective

    if resolved_base_loss == "cov" and not has_group_covariances:
        raise ValueError(
            "transform_search objective requires group_covariances, but they are unavailable."
        )
    return resolved_objective, resolved_base_loss


def format_transform_search_objective(
    objective: str,
    base_loss: str,
    tail_lambda: float,
    tail_bins: int,
    tail_weight_mode: str = "mixed_uniform",
    tail_weight_power: float = 2.0,
) -> str:
    if objective == "jtail":
        return (
            f"J_tail(base={base_loss},lambda={tail_lambda:.3g},bins={tail_bins},"
            f"weights={tail_weight_mode},power={tail_weight_power:.3g})"
        )
    return objective


@torch.no_grad()
def search_best_group_transform(
    weights: Sequence[torch.Tensor],
    group_size: int,
    quantizer_kwargs: dict[str, Any],
    candidates: Sequence[str],
    device: torch.device,
    group_covariances: Sequence[torch.Tensor] | None = None,
    objective: str = "mse",
    base_loss: str = "mse",
    tail_lambda: float = 0.0,
    tail_bins: int = 4,
    tail_weight_mode: str = "mixed_uniform",
    tail_weight_power: float = 2.0,
    auto_default_objective: str = "mse",
) -> MixedGroupTransform:
    if not weights:
        raise ValueError("weights must contain at least one tensor.")

    in_features = weights[0].shape[-1]
    for weight in weights:
        if weight.shape[-1] != in_features:
            raise ValueError("All weights sharing a transform must have the same input feature size.")

    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}.")

    has_group_covariances = group_covariances is not None
    if has_group_covariances and len(group_covariances) != len(weights):
        raise ValueError("group_covariances must be None or have the same length as weights.")
    if tail_lambda < 0:
        raise ValueError(f"tail_lambda must be >= 0, got {tail_lambda}.")
    if tail_bins < 2:
        raise ValueError(f"tail_bins must be >= 2, got {tail_bins}.")
    if tail_weight_power <= 0:
        raise ValueError(f"tail_weight_power must be > 0, got {tail_weight_power}.")
    if tail_weight_mode not in SUPPORTED_TAIL_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported tail_weight_mode '{tail_weight_mode}'. "
            f"Supported: {sorted(SUPPORTED_TAIL_WEIGHT_MODES)}"
        )

    resolved_objective, resolved_base_loss = resolve_transform_search_objective(
        objective=objective,
        base_loss=base_loss,
        has_group_covariances=has_group_covariances,
        auto_default=auto_default_objective,
    )

    candidates = _validate_candidates(candidates)
    if len(candidates) == 0:
        raise ValueError("candidates must contain at least one transform class.")

    # Precompute candidate forward/backward matrices once per search.
    candidate_forward: dict[str, torch.Tensor] = {}
    candidate_backward: dict[str, torch.Tensor] = {}
    active_candidates: list[str] = []
    for name in candidates:
        try:
            transform = build_transform(name, size=group_size, group_size=group_size, device=device, dtype=torch.float32)
            forward_matrix, backward_matrix_from_transform = get_transform_matrices(
                transform,
                size=group_size,
                device=device,
                dtype=torch.float32,
            )

            # Original backward matrix (from transform(inv_t=True)) kept for reference.
            # backward_matrix = backward_matrix_from_transform
            # New strict path: derive backward matrix directly as inverse-transpose of forward matrix.
            # This enforces linear equivalence even if a transform's custom inv_t has implementation drift.
            try:
                backward_matrix = torch.linalg.inv(forward_matrix).T
            except RuntimeError as inv_exc:
                print(f"[transform_search] Skipping candidate '{name}' (non-invertible matrix): {inv_exc}")
                continue

            # Report consistency gap between transform-provided inv_t and strict inverse-transpose.
            transform_pair_gap = (backward_matrix - backward_matrix_from_transform).abs().max().item()
            if transform_pair_gap > 1e-3:
                print(
                    f"[transform_search] Candidate '{name}' inv_t gap max={transform_pair_gap:.2e}; "
                    "using strict inverse-transpose."
                )

            candidate_forward[name] = forward_matrix.to(device=device, dtype=torch.float32)
            candidate_backward[name] = backward_matrix.to(device=device, dtype=torch.float32)
            active_candidates.append(name)
        except Exception as exc:
            # Keep search robust when optional dependencies for a candidate are unavailable.
            print(f"[transform_search] Skipping candidate '{name}': {exc}")

    if len(active_candidates) == 0:
        raise ValueError("No valid transform candidates are available for transform_search.")

    num_groups = in_features // group_size
    selected_names: list[str] = []
    selected_forward: list[torch.Tensor] = []
    selected_backward: list[torch.Tensor] = []
    selected_tail_weight_modes: list[str] | None = [] if resolved_objective == "jtail" else None

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size

        best_name = active_candidates[0]
        best_error = float("inf")

        # New: when mode=auto_abm, classify this group block into A/B/mixed before candidate scoring.
        group_tail_weight_mode = tail_weight_mode
        if resolved_objective == "jtail" and tail_weight_mode == "auto_abm":
            group_weights_for_mode = [w[:, start:end].to(torch.float32) for w in weights]
            group_tail_weight_mode = _select_tail_weight_mode_for_group(group_weights_for_mode)

        for name in active_candidates:
            total_error = 0.0
            inv_t_matrix = candidate_backward[name]
            forward_matrix = candidate_forward[name]

            # Aggregate score over all layers that share this transform slot.
            for weight_idx, weight in enumerate(weights):
                group_weight = weight[:, start:end].to(torch.float32)
                rotated_group = group_weight.matmul(inv_t_matrix)

                delta_w = _compute_quantization_delta(rotated_group, quantizer_kwargs)

                # Original objective branch kept for reference (requested):
                # if group_covariances is None:
                #     total_error += _compute_quantization_mse(rotated_group, quantizer_kwargs)
                # else:
                #     base_covariance = group_covariances[weight_idx][group_idx].to(device=device, dtype=torch.float32)
                #     rotated_covariance = forward_matrix.transpose(-1, -2).matmul(base_covariance).matmul(forward_matrix)
                #     total_error += _compute_gptq_consistent_group_error(
                #         rotated_group_weight=rotated_group,
                #         quantizer_kwargs=quantizer_kwargs,
                #         rotated_group_covariance=rotated_covariance,
                #     )
                if resolved_base_loss == "mse":
                    base_error = delta_w.pow(2).mean().item()
                else:
                    base_covariance = group_covariances[weight_idx][group_idx].to(device=device, dtype=torch.float32)
                    # Because activation is transformed as x' = xT, covariance becomes Cov' = T^T Cov T.
                    rotated_covariance = forward_matrix.transpose(-1, -2).matmul(base_covariance).matmul(forward_matrix)
                    base_error = _compute_gptq_consistent_group_error(
                        rotated_group_weight=rotated_group,
                        quantizer_kwargs=quantizer_kwargs,
                        rotated_group_covariance=rotated_covariance,
                        delta_w=delta_w,
                    )

                if resolved_objective == "jtail":
                    tail_error = _compute_tail_quantile_error(
                        delta_w=delta_w,
                        reference_weight=rotated_group,
                        tail_bins=tail_bins,
                        tail_weight_mode=group_tail_weight_mode,
                        tail_weight_power=tail_weight_power,
                    )
                    total_error += base_error + tail_lambda * tail_error
                else:
                    total_error += base_error

            if total_error < best_error:
                best_error = total_error
                best_name = name

        selected_names.append(best_name)
        selected_forward.append(candidate_forward[best_name])
        selected_backward.append(candidate_backward[best_name])
        if selected_tail_weight_modes is not None:
            selected_tail_weight_modes.append(group_tail_weight_mode)

    return MixedGroupTransform(
        forward_matrices=torch.stack(selected_forward, dim=0),
        backward_matrices=torch.stack(selected_backward, dim=0),
        group_size=group_size,
        selected_transforms=selected_names,
        selected_tail_weight_modes=selected_tail_weight_modes,
    )


@torch.no_grad()
def build_block_input_transforms(
    block,
    hidden_size: int,
    intermediate_size: int,
    args,
    device: torch.device,
    transform_kwargs: dict[str, Any],
    weight_quantizer_kwargs: dict[str, Any] | None,
    slot_input_covariances: dict[str, torch.Tensor] | None = None,
    auto_default_objective: str = "mse",
):
    # Original non-search path preserved.
    if not getattr(args, "transform_search", False):
        qkv_in_transform = build_transform(args.transform_class, size=hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=intermediate_size, **transform_kwargs)
        return qkv_in_transform, o_in_transform, gate_up_in_transform, down_in_transform

    if weight_quantizer_kwargs is None:
        raise ValueError("transform_search requires weight quantization (w_bits < 16).")
    if args.w_granularity != "group" or args.w_group_size is None:
        raise ValueError("transform_search requires --w_granularity group and a valid --w_group_size.")

    candidates = getattr(args, "transform_search_candidates", DEFAULT_SEARCH_TRANSFORMS)
    group_size = args.w_group_size
    resolved_objective, resolved_base_loss = resolve_transform_search_objective(
        objective=getattr(args, "transform_search_objective", "auto"),
        base_loss=getattr(args, "transform_search_base_loss", "cov"),
        has_group_covariances=slot_input_covariances is not None,
        auto_default=auto_default_objective,
    )
    tail_lambda = float(getattr(args, "transform_search_tail_lambda", 0.0))
    tail_bins = int(getattr(args, "transform_search_tail_bins", 4))
    tail_weight_mode = str(getattr(args, "transform_search_tail_weight_mode", "mixed_uniform"))
    tail_weight_power = float(getattr(args, "transform_search_tail_weight_power", 2.0))

    # Search qkv shared transform against q/k/v projection weights.
    qkv_in_transform = search_best_group_transform(
        weights=[block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight],
        group_size=group_size,
        quantizer_kwargs=weight_quantizer_kwargs,
        candidates=candidates,
        device=device,
        group_covariances=(
            [slot_input_covariances["qkv"], slot_input_covariances["qkv"], slot_input_covariances["qkv"]]
            if slot_input_covariances is not None else None
        ),
        objective=resolved_objective,
        base_loss=resolved_base_loss,
        tail_lambda=tail_lambda,
        tail_bins=tail_bins,
        tail_weight_mode=tail_weight_mode,
        tail_weight_power=tail_weight_power,
        auto_default_objective=auto_default_objective,
    )

    # Search o projection input transform.
    o_in_transform = search_best_group_transform(
        weights=[block.self_attn.o_proj.weight],
        group_size=group_size,
        quantizer_kwargs=weight_quantizer_kwargs,
        candidates=candidates,
        device=device,
        group_covariances=([slot_input_covariances["o"]] if slot_input_covariances is not None else None),
        objective=resolved_objective,
        base_loss=resolved_base_loss,
        tail_lambda=tail_lambda,
        tail_bins=tail_bins,
        tail_weight_mode=tail_weight_mode,
        tail_weight_power=tail_weight_power,
        auto_default_objective=auto_default_objective,
    )

    # Search gate/up shared transform against gate/up projection weights.
    gate_up_in_transform = search_best_group_transform(
        weights=[block.mlp.gate_proj.weight, block.mlp.up_proj.weight],
        group_size=group_size,
        quantizer_kwargs=weight_quantizer_kwargs,
        candidates=candidates,
        device=device,
        group_covariances=(
            [slot_input_covariances["gate_up"], slot_input_covariances["gate_up"]]
            if slot_input_covariances is not None else None
        ),
        objective=resolved_objective,
        base_loss=resolved_base_loss,
        tail_lambda=tail_lambda,
        tail_bins=tail_bins,
        tail_weight_mode=tail_weight_mode,
        tail_weight_power=tail_weight_power,
        auto_default_objective=auto_default_objective,
    )

    # Search down projection input transform.
    down_in_transform = search_best_group_transform(
        weights=[block.mlp.down_proj.weight],
        group_size=group_size,
        quantizer_kwargs=weight_quantizer_kwargs,
        candidates=candidates,
        device=device,
        group_covariances=([slot_input_covariances["down"]] if slot_input_covariances is not None else None),
        objective=resolved_objective,
        base_loss=resolved_base_loss,
        tail_lambda=tail_lambda,
        tail_bins=tail_bins,
        tail_weight_mode=tail_weight_mode,
        tail_weight_power=tail_weight_power,
        auto_default_objective=auto_default_objective,
    )

    return qkv_in_transform, o_in_transform, gate_up_in_transform, down_in_transform


def format_transform_summary(transform: BaseTransform) -> str:
    if isinstance(transform, MixedGroupTransform):
        counts = transform.summary()
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        summary = ", ".join([f"{name}:{count}" for name, count in ordered])

        tail_counts = transform.tail_weight_mode_summary()
        if len(tail_counts) > 0:
            tail_ordered = sorted(tail_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            tail_summary = ", ".join([f"{name}:{count}" for name, count in tail_ordered])
            summary = f"{summary} | tail_mode={tail_summary}"
        return summary
    return "single_transform"







@torch.no_grad()
def get_export_transform_matrices(
    transform: BaseTransform,
    layer_in_features: int,
    fallback_group_size: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_groupwise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get compact transform matrices for checkpoint export.

    - For MixedGroupTransform:
      - pseudoquant path can export full per-group bank: [num_groups, g, g]
      - realquant path falls back to one representative block: [g, g]
    - For single transforms:
      - export per-block matrix whenever possible to keep checkpoints compact
    """
    if isinstance(transform, MixedGroupTransform):
        if allow_groupwise:
            return (
                transform.forward_matrices.to(device=device, dtype=dtype),
                transform.backward_matrices.to(device=device, dtype=dtype),
            )

        # Backward-compatible fallback for backends that only accept one matrix.
        # Original behavior exported a single matrix; keep that contract for realquant.
        counts = transform.summary()
        if len(counts) > 1:
            print(
                "[transform_search] Warning: realquant export does not support per-group mixed transforms; "
                "falling back to the most frequent transform block."
            )
        if len(transform.selected_transforms) > 0:
            best_name = max(counts.items(), key=lambda kv: kv[1])[0]
            best_idx = transform.selected_transforms.index(best_name)
        else:
            best_idx = 0
        return (
            transform.forward_matrices[best_idx].to(device=device, dtype=dtype),
            transform.backward_matrices[best_idx].to(device=device, dtype=dtype),
        )

    # Original full-size materialization is kept as a fallback.
    # New default: export compact block matrix size when transform supports it.
    export_size = getattr(transform, "group_size", None) or fallback_group_size or layer_in_features
    try:
        return get_transform_matrices(
            transform,
            size=export_size,
            device=device,
            dtype=dtype,
        )
    except Exception:
        return get_transform_matrices(
            transform,
            size=layer_in_features,
            device=device,
            dtype=dtype,
        )

