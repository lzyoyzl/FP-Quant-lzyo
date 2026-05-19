"""
Usage example:

python analyze_section_3_3_quant_error.py ^
  --model-root /cephfs/shared/model ^
  --models llama-2-7b-hf llama-3-8b-hf Qwen3-8B ^
  --format mxfp ^
  --group-size 32 ^
  --slots o down ^
  --layer-indices 2 16 30 ^
  --num-sequences 64 ^
  --sequence-length 2048 ^
  --dtype bfloat16 ^
  --device cuda ^
  --trust-remote-code ^
  --output-dir ./section_3_3_quant_error_outputs/mxfp_g32

python analyze_section_3_3_quant_error.py ^
  --model-root /cephfs/shared/model ^
  --models llama-2-7b-hf llama-3-8b-hf Qwen3-8B ^
  --format nvfp ^
  --group-size 16 ^
  --slots o down ^
  --layer-indices 2 16 30 ^
  --num-sequences 64 ^
  --sequence-length 2048 ^
  --dtype bfloat16 ^
  --device cuda ^
  --trust-remote-code ^
  --output-dir ./section_3_3_quant_error_outputs/nvfp_g16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.quantizer import Quantizer  # noqa: E402
from src.utils.data_utils import get_data  # noqa: E402


DEFAULT_MODELS = ["llama-2-7b-hf", "llama-3-8b-hf", "Qwen3-8B"]
DEFAULT_SLOTS = ["o", "down"]
DEFAULT_CALIB_ROOT = PROJECT_ROOT / "oyzl_test" / "datasets" / "calib"
MODEL_TO_CALIB_DIR = {
    "llama-2-7b-hf": "fineweb_1024x2048_llama_2_7b_hf",
    "llama-3-8b-hf": "fineweb_1024x2048_llama_3_8b",
    "Qwen3-8B": "fineweb_1024x2048_Qwen3_8B",
}
SLOT_TO_NAME = {
    "qkv": "qkv input",
    "o": "o_proj input",
    "gate_up": "gate/up input",
    "down": "down_proj input",
}


@dataclass
class QuantizationConfig:
    fmt: str
    group_size: int
    observer: str

    @property
    def scale_precision(self) -> str:
        if self.fmt == "nvfp":
            return "e4m3"
        if self.fmt == "mxfp":
            return "e8m0"
        raise ValueError(f"Unsupported format: {self.fmt}")


class GroupErrorAccumulator:
    def __init__(self, hidden_dim: int, group_size: int) -> None:
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.num_groups = math.ceil(hidden_dim / group_size)
        self.valid_widths = [group_size] * self.num_groups
        tail = hidden_dim % group_size
        if tail:
            self.valid_widths[-1] = tail

        self.num_rows = 0
        self.sum_group_mse = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_group_mae = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_group_max_abs = torch.zeros(self.num_groups, dtype=torch.float64)

    def update(self, x_grouped: torch.Tensor, dq_grouped: torch.Tensor) -> None:
        err = (x_grouped - dq_grouped).to(torch.float32)
        group_mse = err.square().mean(dim=-1).to(torch.float64)
        group_mae = err.abs().mean(dim=-1).to(torch.float64)
        group_max_abs = x_grouped.abs().max(dim=-1).values.to(torch.float64)
        self.sum_group_mse += group_mse.sum(dim=0).cpu()
        self.sum_group_mae += group_mae.sum(dim=0).cpu()
        self.sum_group_max_abs += group_max_abs.sum(dim=0).cpu()
        self.num_rows += x_grouped.shape[0]

    def rows(self, model: str, layer_idx: int, slot: str, fmt: str) -> list[dict[str, object]]:
        if self.num_rows == 0:
            return []
        rows = []
        avg_mse = self.sum_group_mse / self.num_rows
        avg_mae = self.sum_group_mae / self.num_rows
        avg_max = self.sum_group_max_abs / self.num_rows
        order = torch.argsort(avg_mse, descending=True)
        rank_map = {int(group_idx.item()): rank + 1 for rank, group_idx in enumerate(order)}
        for group_idx in range(self.num_groups):
            rows.append(
                {
                    "model": model,
                    "layer_idx": layer_idx,
                    "slot": slot,
                    "slot_display": SLOT_TO_NAME[slot],
                    "format": fmt,
                    "group_size": self.group_size,
                    "group_idx": group_idx,
                    "group_rank_by_mse": rank_map[group_idx],
                    "group_start": group_idx * self.group_size,
                    "group_end": min((group_idx + 1) * self.group_size, self.hidden_dim),
                    "valid_width": self.valid_widths[group_idx],
                    "num_token_rows": self.num_rows,
                    "avg_group_mse": float(avg_mse[group_idx].item()),
                    "avg_group_mae": float(avg_mae[group_idx].item()),
                    "avg_group_max_abs": float(avg_max[group_idx].item()),
                }
            )
        return rows


class TopGroupBinAccumulator:
    def __init__(
        self,
        hidden_dim: int,
        group_size: int,
        top_group_indices: list[int],
        num_bins: int,
        low_bin_upper: float,
        high_bin_lower: float,
        dominance_threshold: float,
        dominance_gap: float,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.top_group_indices = list(top_group_indices)
        self.num_bins = num_bins
        self.low_bin_upper = low_bin_upper
        self.high_bin_lower = high_bin_lower
        self.dominance_threshold = dominance_threshold
        self.dominance_gap = dominance_gap
        self.valid_widths = {}
        for group_idx in self.top_group_indices:
            start = group_idx * group_size
            end = min((group_idx + 1) * group_size, hidden_dim)
            self.valid_widths[group_idx] = end - start
        self.error_sums = {group_idx: torch.zeros(num_bins, dtype=torch.float64) for group_idx in self.top_group_indices}
        self.value_counts = {group_idx: torch.zeros(num_bins, dtype=torch.float64) for group_idx in self.top_group_indices}
        self.total_error = {group_idx: 0.0 for group_idx in self.top_group_indices}
        self.total_values = {group_idx: 0.0 for group_idx in self.top_group_indices}

    def update(self, x_grouped: torch.Tensor, dq_grouped: torch.Tensor) -> None:
        err_sq = (x_grouped - dq_grouped).to(torch.float32).square()
        for group_idx in self.top_group_indices:
            valid_width = self.valid_widths[group_idx]
            gx = x_grouped[:, group_idx, :valid_width].abs().to(torch.float32)
            ge = err_sq[:, group_idx, :valid_width]
            row_max = gx.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
            norm = gx / row_max
            bins = torch.clamp((norm * self.num_bins).long(), max=self.num_bins - 1).reshape(-1)
            flat_err = ge.reshape(-1).to(torch.float64).cpu()
            ones = torch.ones_like(flat_err, dtype=torch.float64)

            err_acc = torch.zeros(self.num_bins, dtype=torch.float64)
            cnt_acc = torch.zeros(self.num_bins, dtype=torch.float64)
            err_acc.scatter_add_(0, bins.cpu(), flat_err)
            cnt_acc.scatter_add_(0, bins.cpu(), ones)

            self.error_sums[group_idx] += err_acc
            self.value_counts[group_idx] += cnt_acc
            self.total_error[group_idx] += float(flat_err.sum().item())
            self.total_values[group_idx] += float(ones.sum().item())

    def classify(self, error_ratios: torch.Tensor) -> tuple[str, str, float, float]:
        centers = torch.linspace(0.5 / self.num_bins, 1.0 - 0.5 / self.num_bins, self.num_bins)
        low_share = float(error_ratios[centers <= self.low_bin_upper].sum().item())
        high_share = float(error_ratios[centers >= self.high_bin_lower].sum().item())
        if (
            low_share >= self.dominance_threshold
            and low_share >= high_share + self.dominance_gap
        ):
            return "outlier_squeezing", "离群值挤压型", low_share, high_share
        if (
            high_share >= self.dominance_threshold
            and high_share >= low_share + self.dominance_gap
        ):
            return "large_value_dominant", "大值主导型", low_share, high_share
        return "mixed", "混合型", low_share, high_share

    def rows(
        self,
        model: str,
        layer_idx: int,
        slot: str,
        fmt: str,
        rank_map: dict[int, int],
        group_rows_map: dict[int, dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        bin_edges = torch.linspace(0.0, 1.0, self.num_bins + 1)
        for group_idx in self.top_group_indices:
            err_total = max(self.total_error[group_idx], 1e-12)
            val_total = max(self.total_values[group_idx], 1e-12)
            err_ratio = self.error_sums[group_idx] / err_total
            val_ratio = self.value_counts[group_idx] / val_total
            pattern_code, pattern_label_zh, low_share, high_share = self.classify(err_ratio)
            group_meta = group_rows_map[group_idx]
            summary_rows.append(
                {
                    "model": model,
                    "layer_idx": layer_idx,
                    "slot": slot,
                    "slot_display": SLOT_TO_NAME[slot],
                    "format": fmt,
                    "group_size": self.group_size,
                    "top_rank": rank_map[group_idx],
                    "group_idx": group_idx,
                    "avg_group_mse": group_meta["avg_group_mse"],
                    "avg_group_max_abs": group_meta["avg_group_max_abs"],
                    "pattern_code": pattern_code,
                    "pattern_label_zh": pattern_label_zh,
                    "low_bin_error_share": low_share,
                    "high_bin_error_share": high_share,
                }
            )
            for bin_idx in range(self.num_bins):
                detail_rows.append(
                    {
                        "model": model,
                        "layer_idx": layer_idx,
                        "slot": slot,
                        "slot_display": SLOT_TO_NAME[slot],
                        "format": fmt,
                        "group_size": self.group_size,
                        "top_rank": rank_map[group_idx],
                        "group_idx": group_idx,
                        "pattern_code": pattern_code,
                        "pattern_label_zh": pattern_label_zh,
                        "bin_idx": bin_idx,
                        "bin_left": float(bin_edges[bin_idx].item()),
                        "bin_right": float(bin_edges[bin_idx + 1].item()),
                        "error_contribution": float(self.error_sums[group_idx][bin_idx].item()),
                        "error_contribution_ratio": float(err_ratio[bin_idx].item()),
                        "value_count": float(self.value_counts[group_idx][bin_idx].item()),
                        "value_count_ratio": float(val_ratio[bin_idx].item()),
                    }
                )
        return detail_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze group-wise activation quantization error sources for thesis Section 3.3."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--calib-root", type=str, default=str(DEFAULT_CALIB_ROOT))
    parser.add_argument("--dataset-name-or-path", type=str, default="fineweb-edu")
    parser.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS, choices=["qkv", "o", "gate_up", "down"])
    parser.add_argument("--layer-indices", nargs="+", type=int, default=None)
    parser.add_argument("--num-sampled-layers", type=int, default=3)
    parser.add_argument("--format", choices=["mxfp", "nvfp"], default="mxfp")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--observer", choices=["minmax", "mse"], default="minmax")
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--top-k-groups", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--low-bin-upper", type=float, default=0.3)
    parser.add_argument("--high-bin-lower", type=float, default=0.7)
    parser.add_argument("--dominance-threshold", type=float, default=0.55)
    parser.add_argument("--dominance-gap", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THIS_DIR / "section_3_3_quant_error_outputs"),
    )
    return parser.parse_args()


def infer_group_size(fmt: str, explicit_group_size: int | None) -> int:
    if explicit_group_size is not None:
        return explicit_group_size
    return 16 if fmt == "nvfp" else 32


def resolve_model_path(model_name: str, model_root: str | None) -> str:
    if model_root is None:
        return model_name
    candidate = Path(model_root) / model_name
    return str(candidate if candidate.exists() else model_name)


def resolve_calibration_path(model_name: str, dataset_name_or_path: str, calib_root: str | None) -> str:
    explicit_path = Path(dataset_name_or_path)
    if explicit_path.is_file():
        return str(explicit_path)

    if calib_root is None:
        return dataset_name_or_path
    calib_dir_name = MODEL_TO_CALIB_DIR.get(model_name)
    if calib_dir_name is None:
        return dataset_name_or_path

    calib_dir = Path(calib_root) / calib_dir_name
    token_path = calib_dir / "fineweb_calib_1024x2048_tokens.pt"
    text_path = calib_dir / "fineweb_calib_1024x2048_text.jsonl"
    if token_path.exists():
        return str(token_path)
    if text_path.exists():
        return str(text_path)
    return dataset_name_or_path


def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def reshape_hook_input(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim >= 3:
        return matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim == 2:
        return matrix
    raise ValueError(f"Unsupported activation shape: {tuple(matrix.shape)}")


def select_layers(num_layers: int, explicit_indices: list[int] | None, num_sampled_layers: int) -> list[int]:
    if explicit_indices:
        return sorted({idx for idx in explicit_indices if 0 <= idx < num_layers})
    if num_sampled_layers >= num_layers:
        return list(range(num_layers))
    if num_sampled_layers <= 1:
        return [num_layers // 2]
    indices = []
    for i in range(num_sampled_layers):
        idx = round(i * (num_layers - 1) / (num_sampled_layers - 1))
        indices.append(idx)
    return sorted(set(indices))


def quantize_matrix_groupwise(matrix: torch.Tensor, qcfg: QuantizationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    original_dim = matrix.shape[-1]
    pad_cols = (-original_dim) % qcfg.group_size
    if pad_cols:
        matrix = F.pad(matrix, (0, pad_cols))
    quantizer = Quantizer(
        bits=4,
        symmetric=True,
        format=qcfg.fmt,
        granularity="group",
        observer=qcfg.observer,
        dim=-1,
        group_size=qcfg.group_size,
        scale_precision=qcfg.scale_precision,
    )
    scales, zeros = quantizer.get_quantization_params(matrix)
    dq = quantizer(matrix, scales, zeros)
    num_groups = matrix.shape[-1] // qcfg.group_size
    grouped_x = matrix.view(matrix.shape[0], num_groups, qcfg.group_size)
    grouped_dq = dq.view(dq.shape[0], num_groups, qcfg.group_size)
    return grouped_x, grouped_dq


def get_blocks(model) -> Iterable:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Unsupported model structure: expected model.model.layers")


def build_group_error_hooks(blocks, selected_layers, slots, accumulators, qcfg: QuantizationConfig):
    hooks = []
    selected_set = set(selected_layers)

    for layer_idx, block in enumerate(blocks):
        if layer_idx not in selected_set:
            continue

        def make_hook(slot_name: str, cur_layer_idx: int):
            def _hook(_, inp, __):
                if not inp:
                    return
                matrix = reshape_hook_input(inp[0]).detach().to(dtype=torch.float32)
                grouped_x, grouped_dq = quantize_matrix_groupwise(matrix, qcfg)
                accumulators[cur_layer_idx][slot_name].update(grouped_x, grouped_dq)

            return _hook

        if "qkv" in slots:
            hooks.append(block.self_attn.q_proj.register_forward_hook(make_hook("qkv", layer_idx)))
        if "o" in slots:
            hooks.append(block.self_attn.o_proj.register_forward_hook(make_hook("o", layer_idx)))
        if "gate_up" in slots:
            hooks.append(block.mlp.gate_proj.register_forward_hook(make_hook("gate_up", layer_idx)))
        if "down" in slots:
            hooks.append(block.mlp.down_proj.register_forward_hook(make_hook("down", layer_idx)))
    return hooks


def build_top_group_hooks(blocks, selected_layers, slots, accumulators, qcfg: QuantizationConfig):
    hooks = []
    selected_set = set(selected_layers)

    for layer_idx, block in enumerate(blocks):
        if layer_idx not in selected_set:
            continue

        def make_hook(slot_name: str, cur_layer_idx: int):
            def _hook(_, inp, __):
                if not inp:
                    return
                matrix = reshape_hook_input(inp[0]).detach().to(dtype=torch.float32)
                grouped_x, grouped_dq = quantize_matrix_groupwise(matrix, qcfg)
                accumulators[cur_layer_idx][slot_name].update(grouped_x, grouped_dq)

            return _hook

        if "qkv" in slots:
            hooks.append(block.self_attn.q_proj.register_forward_hook(make_hook("qkv", layer_idx)))
        if "o" in slots:
            hooks.append(block.self_attn.o_proj.register_forward_hook(make_hook("o", layer_idx)))
        if "gate_up" in slots:
            hooks.append(block.mlp.gate_proj.register_forward_hook(make_hook("gate_up", layer_idx)))
        if "down" in slots:
            hooks.append(block.mlp.down_proj.register_forward_hook(make_hook("down", layer_idx)))
    return hooks


def run_forward_pass(model, calibration_data: list[torch.Tensor], device: str) -> None:
    with torch.no_grad():
        for sample in calibration_data:
            model(sample.to(device))


def init_group_error_accumulators(blocks, selected_layers, slots, group_size: int):
    accs = {}
    for layer_idx in selected_layers:
        block = blocks[layer_idx]
        hidden_dims = {
            "qkv": int(block.self_attn.q_proj.weight.shape[-1]),
            "o": int(block.self_attn.o_proj.weight.shape[-1]),
            "gate_up": int(block.mlp.gate_proj.weight.shape[-1]),
            "down": int(block.mlp.down_proj.weight.shape[-1]),
        }
        accs[layer_idx] = {
            slot: GroupErrorAccumulator(hidden_dims[slot], group_size)
            for slot in slots
        }
    return accs


def finalize_case_summary(
    group_rows: list[dict[str, object]],
    model: str,
    layer_idx: int,
    slot: str,
    fmt: str,
    group_size: int,
) -> dict[str, object]:
    case_rows = [row for row in group_rows if row["model"] == model and row["layer_idx"] == layer_idx and row["slot"] == slot]
    mse = torch.tensor([float(row["avg_group_mse"]) for row in case_rows], dtype=torch.float64)
    max_abs = torch.tensor([float(row["avg_group_max_abs"]) for row in case_rows], dtype=torch.float64)
    if mse.numel() > 1:
        corr = torch.corrcoef(torch.stack([mse, max_abs], dim=0))[0, 1].item()
    else:
        corr = float("nan")
    top_groups = sorted(case_rows, key=lambda row: float(row["avg_group_mse"]), reverse=True)[:5]
    return {
        "model": model,
        "layer_idx": layer_idx,
        "slot": slot,
        "slot_display": SLOT_TO_NAME[slot],
        "format": fmt,
        "group_size": group_size,
        "num_groups": len(case_rows),
        "num_token_rows": int(case_rows[0]["num_token_rows"]) if case_rows else 0,
        "mean_group_mse": float(mse.mean().item()) if case_rows else 0.0,
        "max_group_mse": float(mse.max().item()) if case_rows else 0.0,
        "mean_group_max_abs": float(max_abs.mean().item()) if case_rows else 0.0,
        "pearson_group_mse_vs_max_abs": float(corr),
        "top_group_indices": ",".join(str(int(row["group_idx"])) for row in top_groups),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = infer_group_size(args.format, args.group_size)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_torch_dtype(args.dtype)
    qcfg = QuantizationConfig(fmt=args.format, group_size=group_size, observer=args.observer)

    all_group_rows: list[dict[str, object]] = []
    all_case_rows: list[dict[str, object]] = []
    all_top_group_rows: list[dict[str, object]] = []
    all_bin_rows: list[dict[str, object]] = []

    meta = {
        "models": args.models,
        "slots": args.slots,
        "format": args.format,
        "group_size": group_size,
        "observer": args.observer,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "layer_indices": args.layer_indices,
        "num_sampled_layers": args.num_sampled_layers,
        "top_k_groups": args.top_k_groups,
        "num_bins": args.num_bins,
        "low_bin_upper": args.low_bin_upper,
        "high_bin_lower": args.high_bin_lower,
        "dominance_threshold": args.dominance_threshold,
        "dominance_gap": args.dominance_gap,
        "device": device,
        "dtype": args.dtype,
    }

    for model_name in args.models:
        model_path = resolve_model_path(model_name, args.model_root)
        print(f"[INFO] Loading model: {model_name} <- {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=None,
        )
        model.to(device)
        model.eval()

        calibration_source = resolve_calibration_path(model_name, args.dataset_name_or_path, args.calib_root)
        print(f"[INFO] Calibration source for {model_name}: {calibration_source}")
        calibration_data = get_data(
            calibration_source,
            tokenizer,
            args.sequence_length,
            num_calibration_samples=args.num_sequences,
            seed=args.seed,
        )

        blocks = list(get_blocks(model))
        selected_layers = select_layers(len(blocks), args.layer_indices, args.num_sampled_layers)
        print(f"[INFO] Selected layers for {model_name}: {selected_layers}")

        first_pass_accs = init_group_error_accumulators(blocks, selected_layers, args.slots, group_size)
        hooks = build_group_error_hooks(blocks, selected_layers, args.slots, first_pass_accs, qcfg)
        try:
            run_forward_pass(model, calibration_data, device)
        finally:
            for hook in hooks:
                hook.remove()

        model_group_rows: list[dict[str, object]] = []
        group_rows_map: dict[tuple[int, str], dict[int, dict[str, object]]] = {}
        top_groups_by_case: dict[tuple[int, str], list[int]] = {}
        for layer_idx in selected_layers:
            group_rows_map[(layer_idx, "")] = {}
            for slot in args.slots:
                rows = first_pass_accs[layer_idx][slot].rows(model_name, layer_idx, slot, args.format)
                model_group_rows.extend(rows)
                by_group = {int(row["group_idx"]): row for row in rows}
                group_rows_map[(layer_idx, slot)] = by_group
                top_groups = sorted(rows, key=lambda row: float(row["avg_group_mse"]), reverse=True)[: args.top_k_groups]
                top_groups_by_case[(layer_idx, slot)] = [int(row["group_idx"]) for row in top_groups]
                all_case_rows.append(finalize_case_summary(rows, model_name, layer_idx, slot, args.format, group_size))
        all_group_rows.extend(model_group_rows)

        second_pass_accs = {}
        for layer_idx in selected_layers:
            second_pass_accs[layer_idx] = {}
            for slot in args.slots:
                hidden_dim = first_pass_accs[layer_idx][slot].hidden_dim
                second_pass_accs[layer_idx][slot] = TopGroupBinAccumulator(
                    hidden_dim=hidden_dim,
                    group_size=group_size,
                    top_group_indices=top_groups_by_case[(layer_idx, slot)],
                    num_bins=args.num_bins,
                    low_bin_upper=args.low_bin_upper,
                    high_bin_lower=args.high_bin_lower,
                    dominance_threshold=args.dominance_threshold,
                    dominance_gap=args.dominance_gap,
                )

        hooks = build_top_group_hooks(blocks, selected_layers, args.slots, second_pass_accs, qcfg)
        try:
            run_forward_pass(model, calibration_data, device)
        finally:
            for hook in hooks:
                hook.remove()

        for layer_idx in selected_layers:
            for slot in args.slots:
                rank_map = {group_idx: rank + 1 for rank, group_idx in enumerate(top_groups_by_case[(layer_idx, slot)])}
                detail_rows, summary_rows = second_pass_accs[layer_idx][slot].rows(
                    model=model_name,
                    layer_idx=layer_idx,
                    slot=slot,
                    fmt=args.format,
                    rank_map=rank_map,
                    group_rows_map=group_rows_map[(layer_idx, slot)],
                )
                all_bin_rows.extend(detail_rows)
                all_top_group_rows.extend(summary_rows)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output_dir / "group_error_stats.csv", all_group_rows)
    write_csv(output_dir / "case_summary.csv", all_case_rows)
    write_csv(output_dir / "top_group_summary.csv", all_top_group_rows)
    write_csv(output_dir / "top_group_bin_contrib.csv", all_bin_rows)
    (output_dir / "analysis_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
