'''
python analyze_section_3_2_distribution.py 
  --model-root /cephfs/shared/model 
  --models llama-2-7b-hf llama-3-8b-hf Qwen3-8B 
  --group-sizes 16 32 
  --num-sequences 1024 
  --sequence-length 2048 
  --num-sampled-layers 8
  --outlier-quantile 0.999 
  --dtype bfloat16 
  --device cuda 
  --trust-remote-code 
  --output-dir ./section_3_2_outputs
'''
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_utils import get_data  # noqa: E402


DEFAULT_MODELS = ["llama-2-7b-hf", "llama-3-8b-hf", "Qwen3-8B"]
DEFAULT_GROUP_SIZES = [16, 32]
DEFAULT_SLOTS = ["qkv", "o", "gate_up", "down"]
DEFAULT_CALIB_ROOT = PROJECT_ROOT / "oyzl_test" / "datasets" / "calib"
MODEL_TO_CALIB_DIR = {
    "llama-2-7b-hf": "fineweb_1024x2048_llama_2_7b_hf",
    "llama-3-8b-hf": "fineweb_1024x2048_llama_3_8b",
    "Qwen3-8B": "fineweb_1024x2048_Qwen3_8B",
}


@dataclass
class ActivationValueSampler:
    sample_size_per_call: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.samples: list[torch.Tensor] = []

    def update_from_matrix(self, matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D activation matrix, got shape={tuple(matrix.shape)}")
        flat = matrix.detach().abs().reshape(-1).to(device="cpu", dtype=torch.float32)
        if flat.numel() == 0:
            return
        take = min(self.sample_size_per_call, flat.numel())
        if take == flat.numel():
            sample = flat
        else:
            idx = torch.tensor(
                self.rng.sample(range(flat.numel()), k=take),
                dtype=torch.long,
            )
            sample = flat[idx]
        self.samples.append(sample)

    def finalize_quantile(self, quantile: float) -> float:
        if not self.samples:
            raise ValueError("No activation samples collected for quantile estimation.")
        values = torch.cat(self.samples, dim=0)
        return float(torch.quantile(values, q=quantile).item())


@dataclass
class GroupActivationAccumulator:
    hidden_dim: int
    group_size: int
    outlier_threshold: float
    outlier_quantile: float
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.num_groups = (self.hidden_dim + self.group_size - 1) // self.group_size
        self.valid_widths = torch.full((self.num_groups,), self.group_size, dtype=torch.float64)
        tail = self.hidden_dim % self.group_size
        if tail != 0:
            self.valid_widths[-1] = tail

        self.num_token_rows = torch.zeros(self.num_groups, dtype=torch.float64)
        self.total_values = torch.zeros(self.num_groups, dtype=torch.float64)
        self.outlier_values = torch.zeros(self.num_groups, dtype=torch.float64)
        self.tokens_with_any_outlier = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_mean_abs = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_rms = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_max_over_rms = torch.zeros(self.num_groups, dtype=torch.float64)
        self.sum_top1_energy_share = torch.zeros(self.num_groups, dtype=torch.float64)
        self.max_abs = torch.zeros(self.num_groups, dtype=torch.float64)
        self.channel_has_outlier = torch.zeros((self.num_groups, self.group_size), dtype=torch.bool)

    def update_from_matrix(self, matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D activation matrix, got shape={tuple(matrix.shape)}")
        matrix = matrix.detach().abs().to(device="cpu", dtype=torch.float32)
        rows, cols = matrix.shape
        if cols != self.hidden_dim:
            raise ValueError(f"Hidden dim mismatch: expected {self.hidden_dim}, got {cols}")

        pad_cols = self.num_groups * self.group_size - cols
        if pad_cols > 0:
            matrix = torch.nn.functional.pad(matrix, (0, pad_cols))

        grouped = matrix.view(rows, self.num_groups, self.group_size)
        mean_abs = grouped.mean(dim=-1).to(torch.float64)
        rms = grouped.square().mean(dim=-1).sqrt().to(torch.float64)
        max_abs = grouped.max(dim=-1).values.to(torch.float64)
        energy = grouped.square().sum(dim=-1).to(torch.float64)
        top1_energy_share = max_abs.square() / (energy + self.eps)
        max_over_rms = max_abs / (rms + self.eps)

        outlier_mask = grouped >= self.outlier_threshold
        outlier_count = outlier_mask.sum(dim=(0, 2)).to(torch.float64)
        token_with_any_outlier = outlier_mask.any(dim=-1).sum(dim=0).to(torch.float64)
        channel_any = outlier_mask.any(dim=0)

        self.num_token_rows += rows
        self.total_values += rows * self.valid_widths
        self.outlier_values += outlier_count
        self.tokens_with_any_outlier += token_with_any_outlier
        self.sum_mean_abs += mean_abs.sum(dim=0)
        self.sum_rms += rms.sum(dim=0)
        self.sum_max_over_rms += max_over_rms.sum(dim=0)
        self.sum_top1_energy_share += top1_energy_share.sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, max_abs.max(dim=0).values)
        self.channel_has_outlier |= channel_any

    def rows(self, model_id: str, layer_idx: int, slot_name: str) -> list[dict[str, float | int | str]]:
        baseline_outlier_ratio = 1.0 - self.outlier_quantile
        rows: list[dict[str, float | int | str]] = []
        for group_idx in range(self.num_groups):
            token_rows = float(self.num_token_rows[group_idx].item())
            if token_rows <= 0:
                continue
            valid_width = int(self.valid_widths[group_idx].item())
            start = group_idx * self.group_size
            end = min((group_idx + 1) * self.group_size, self.hidden_dim)
            outlier_value_ratio = self.outlier_values[group_idx].item() / self.total_values[group_idx].item()
            rows.append(
                {
                    "model": model_id,
                    "layer_idx": layer_idx,
                    "slot": slot_name,
                    "group_size": self.group_size,
                    "group_idx": group_idx,
                    "group_start": start,
                    "group_end": end,
                    "num_token_rows": int(round(token_rows)),
                    "outlier_quantile": self.outlier_quantile,
                    "outlier_threshold": self.outlier_threshold,
                    "global_outlier_ratio": baseline_outlier_ratio,
                    "avg_mean_abs": self.sum_mean_abs[group_idx].item() / token_rows,
                    "avg_rms": self.sum_rms[group_idx].item() / token_rows,
                    "avg_max_over_rms": self.sum_max_over_rms[group_idx].item() / token_rows,
                    "avg_top1_energy_share": self.sum_top1_energy_share[group_idx].item() / token_rows,
                    "max_abs": self.max_abs[group_idx].item(),
                    "outlier_value_ratio": outlier_value_ratio,
                    "outlier_density_multiplier": outlier_value_ratio / max(baseline_outlier_ratio, self.eps),
                    "outlier_token_ratio": self.tokens_with_any_outlier[group_idx].item() / token_rows,
                    "channels_with_outlier_fraction": float(
                        self.channel_has_outlier[group_idx, :valid_width].sum().item() / max(valid_width, 1)
                    ),
                }
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sampled-layer group-wise activation distribution statistics for thesis Section 3.2."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=DEFAULT_GROUP_SIZES)
    parser.add_argument("--slots", nargs="+", default=DEFAULT_SLOTS, choices=DEFAULT_SLOTS)
    parser.add_argument("--dataset-name-or-path", type=str, default="fineweb-edu")
    parser.add_argument("--calib-root", type=str, default=str(DEFAULT_CALIB_ROOT))
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--layer-indices", nargs="+", type=int, default=None)
    parser.add_argument("--num-sampled-layers", type=int, default=8)
    parser.add_argument("--outlier-quantile", type=float, default=0.999)
    parser.add_argument("--threshold-sample-size-per-call", type=int, default=4096)
    parser.add_argument("--strong-outlier-multiplier", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default=str(THIS_DIR / "section_3_2_outputs"))
    return parser.parse_args()


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

    mapped_dir = MODEL_TO_CALIB_DIR.get(model_name)
    if mapped_dir is None:
        return dataset_name_or_path

    calib_dir = Path(calib_root) / mapped_dir
    token_path = calib_dir / "fineweb_calib_1024x2048_tokens.pt"
    text_path = calib_dir / "fineweb_calib_1024x2048_text.jsonl"
    if token_path.exists():
        return str(token_path)
    if text_path.exists():
        return str(text_path)
    return dataset_name_or_path


def reshape_hook_input(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 3:
        return matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim > 3:
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


def init_threshold_samplers(selected_layers: list[int], slots: list[str], sample_size_per_call: int, seed: int):
    samplers = {}
    for layer_idx in selected_layers:
        samplers[layer_idx] = {}
        for slot in slots:
            samplers[layer_idx][slot] = ActivationValueSampler(
                sample_size_per_call=sample_size_per_call,
                rng=random.Random(seed * 1000 + layer_idx * 10 + len(slot)),
            )
    return samplers


def init_group_accumulators(blocks, selected_layers: list[int], group_sizes: list[int], slots: list[str], thresholds, outlier_quantile: float):
    accumulators = {}
    for layer_idx in selected_layers:
        block = blocks[layer_idx]
        hidden_dims = {
            "qkv": int(block.self_attn.q_proj.weight.shape[-1]),
            "o": int(block.self_attn.o_proj.weight.shape[-1]),
            "gate_up": int(block.mlp.gate_proj.weight.shape[-1]),
            "down": int(block.mlp.down_proj.weight.shape[-1]),
        }
        accumulators[layer_idx] = {}
        for slot in slots:
            accumulators[layer_idx][slot] = {
                g: GroupActivationAccumulator(
                    hidden_dim=hidden_dims[slot],
                    group_size=g,
                    outlier_threshold=thresholds[layer_idx][slot],
                    outlier_quantile=outlier_quantile,
                )
                for g in group_sizes
            }
    return accumulators


def build_value_sampling_hooks(blocks, selected_layers: list[int], slots: list[str], samplers):
    hooks = []
    selected_set = set(selected_layers)
    for layer_idx, block in enumerate(blocks):
        if layer_idx not in selected_set:
            continue

        def make_hook(slot_name: str, cur_layer_idx: int):
            def _hook(_, inp, out):
                if not inp:
                    return
                matrix = reshape_hook_input(inp[0])
                samplers[cur_layer_idx][slot_name].update_from_matrix(matrix)

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


def build_group_stat_hooks(blocks, selected_layers: list[int], slots: list[str], accumulators, group_sizes: list[int]):
    hooks = []
    selected_set = set(selected_layers)
    for layer_idx, block in enumerate(blocks):
        if layer_idx not in selected_set:
            continue

        def make_hook(slot_name: str, cur_layer_idx: int):
            def _hook(_, inp, out):
                if not inp:
                    return
                matrix = reshape_hook_input(inp[0])
                for group_size in group_sizes:
                    accumulators[cur_layer_idx][slot_name][group_size].update_from_matrix(matrix)

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


def run_forward_pass(model, calibration_data, device: str) -> None:
    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            model(sample)


def finalize_thresholds(samplers, outlier_quantile: float):
    thresholds = {}
    for layer_idx, slot_map in samplers.items():
        thresholds[layer_idx] = {}
        for slot_name, sampler in slot_map.items():
            thresholds[layer_idx][slot_name] = sampler.finalize_quantile(outlier_quantile)
    return thresholds


def finalize_raw_rows(model_name: str, accumulators) -> list[dict[str, float | int | str]]:
    rows = []
    for layer_idx, slot_map in accumulators.items():
        for slot_name, group_map in slot_map.items():
            for accumulator in group_map.values():
                rows.extend(accumulator.rows(model_id=model_name, layer_idx=layer_idx, slot_name=slot_name))
    return rows


def build_layer_summary(raw_rows: list[dict[str, float | int | str]], strong_outlier_multiplier: float) -> list[dict[str, float | int | str]]:
    grouped = {}
    for row in raw_rows:
        key = (
            row["model"],
            row["layer_idx"],
            row["slot"],
            row["group_size"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (model, layer_idx, slot, group_size), rows in grouped.items():
        multipliers = sorted(float(r["outlier_density_multiplier"]) for r in rows)
        token_ratios = sorted(float(r["outlier_token_ratio"]) for r in rows)
        channels_frac = sorted(float(r["channels_with_outlier_fraction"]) for r in rows)
        max_over_rms = sorted(float(r["avg_max_over_rms"]) for r in rows)
        num_groups = len(rows)

        def quantile(sorted_vals: list[float], q: float) -> float:
            if not sorted_vals:
                return float("nan")
            pos = q * (len(sorted_vals) - 1)
            lo = math.floor(pos)
            hi = math.ceil(pos)
            if lo == hi:
                return sorted_vals[lo]
            alpha = pos - lo
            return sorted_vals[lo] * (1.0 - alpha) + sorted_vals[hi] * alpha

        summary_rows.append(
            {
                "model": model,
                "layer_idx": layer_idx,
                "slot": slot,
                "group_size": group_size,
                "num_groups": num_groups,
                "outlier_quantile": rows[0]["outlier_quantile"],
                "outlier_threshold": rows[0]["outlier_threshold"],
                "global_outlier_ratio": rows[0]["global_outlier_ratio"],
                "fraction_groups_density_le_1": sum(m <= 1.0 for m in multipliers) / num_groups,
                "fraction_groups_density_gt_1": sum(m > 1.0 for m in multipliers) / num_groups,
                "fraction_groups_density_gt_strong": sum(m > strong_outlier_multiplier for m in multipliers) / num_groups,
                "median_density_multiplier": quantile(multipliers, 0.5),
                "p90_density_multiplier": quantile(multipliers, 0.9),
                "max_density_multiplier": multipliers[-1],
                "median_outlier_token_ratio": quantile(token_ratios, 0.5),
                "p90_outlier_token_ratio": quantile(token_ratios, 0.9),
                "median_channels_with_outlier_fraction": quantile(channels_frac, 0.5),
                "p90_channels_with_outlier_fraction": quantile(channels_frac, 0.9),
                "median_max_over_rms": quantile(max_over_rms, 0.5),
                "p90_max_over_rms": quantile(max_over_rms, 0.9),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    all_raw_rows: list[dict[str, float | int | str]] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    metadata = {
        "models": args.models,
        "group_sizes": args.group_sizes,
        "slots": args.slots,
        "dataset_name_or_path": args.dataset_name_or_path,
        "calib_root": args.calib_root,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "device": device,
        "dtype": args.dtype,
        "outlier_quantile": args.outlier_quantile,
        "num_sampled_layers": args.num_sampled_layers,
        "strong_outlier_multiplier": args.strong_outlier_multiplier,
    }

    for model_name in args.models:
        model_path = resolve_model_path(model_name, args.model_root)
        print(f"[INFO] Loading model: {model_name} <- {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
        model = model.to(device)
        model.eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
        calib_path = resolve_calibration_path(model_name, args.dataset_name_or_path, args.calib_root)
        print(f"[INFO] Calibration source for {model_name}: {calib_path}")

        calibration_data = get_data(
            calib_path,
            tokenizer,
            args.sequence_length,
            args.num_sequences,
            args.seed,
        )

        blocks = model.model.layers
        selected_layers = select_layers(
            num_layers=len(blocks),
            explicit_indices=args.layer_indices,
            num_sampled_layers=args.num_sampled_layers,
        )
        print(f"[INFO] Selected layers for {model_name}: {selected_layers}")

        samplers = init_threshold_samplers(
            selected_layers=selected_layers,
            slots=args.slots,
            sample_size_per_call=args.threshold_sample_size_per_call,
            seed=args.seed,
        )
        hooks = build_value_sampling_hooks(blocks, selected_layers, args.slots, samplers)
        try:
            run_forward_pass(model, calibration_data, device)
        finally:
            for hook in hooks:
                hook.remove()

        thresholds = finalize_thresholds(samplers, outlier_quantile=args.outlier_quantile)
        for layer_idx in selected_layers:
            for slot_name in args.slots:
                threshold_rows.append(
                    {
                        "model": model_name,
                        "layer_idx": layer_idx,
                        "slot": slot_name,
                        "outlier_quantile": args.outlier_quantile,
                        "outlier_threshold": thresholds[layer_idx][slot_name],
                    }
                )

        accumulators = init_group_accumulators(
            blocks=blocks,
            selected_layers=selected_layers,
            group_sizes=args.group_sizes,
            slots=args.slots,
            thresholds=thresholds,
            outlier_quantile=args.outlier_quantile,
        )
        hooks = build_group_stat_hooks(blocks, selected_layers, args.slots, accumulators, args.group_sizes)
        try:
            run_forward_pass(model, calibration_data, device)
        finally:
            for hook in hooks:
                hook.remove()

        all_raw_rows.extend(finalize_raw_rows(model_name, accumulators))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_summary_rows = build_layer_summary(all_raw_rows, strong_outlier_multiplier=args.strong_outlier_multiplier)

    raw_csv = output_dir / "group_activation_distribution_raw.csv"
    summary_csv = output_dir / "group_activation_distribution_layer_summary.csv"
    threshold_csv = output_dir / "group_activation_outlier_thresholds.csv"
    meta_json = output_dir / "group_activation_distribution_meta.json"
    write_csv(raw_csv, all_raw_rows)
    write_csv(summary_csv, layer_summary_rows)
    write_csv(threshold_csv, threshold_rows)
    meta_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote raw activation statistics to: {raw_csv}")
    print(f"[DONE] Wrote layer summary to: {summary_csv}")
    print(f"[DONE] Wrote outlier thresholds to: {threshold_csv}")
    print(f"[DONE] Wrote metadata to: {meta_json}")


if __name__ == "__main__":
    main()
