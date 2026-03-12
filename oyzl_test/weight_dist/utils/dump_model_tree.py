#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

utils:
python dump_model_tree.py \
  --model_dir /cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct/mxfp_gptq_w4a4_hadamard_h128_mse_default \
  --out_txt llama3_8b_instruct_mxfp_gptq_tree.txt \
  --max_depth 8 \
  --show_shapes \
  --include_params_count




Dump full module tree of an exported FP-Quant model directory,
with depth limit and quantization marks, to a txt file.

Quantization marking is derived from safetensors index keys:
- pseudoquant: *.dqweight
- realquant : *.qweight (and usually *.scales)

Marks:
  [Q]  : module itself is quantized (exact match to a dqweight/qweight prefix)
  [Q+] : module is not directly quantized but has quantized descendants
  [FP] : no quantized module in this subtree

This script builds model skeleton from config only (no checkpoint load),
so it's light-weight and safe for structure inspection.
"""

import os
import json
import re
import argparse
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM


# -----------------------------
# Quantized modules discovery
# -----------------------------
def load_quant_prefixes_from_index(model_dir: str) -> Tuple[Set[str], Set[str], Dict[int, Set[str]], Dict]:
    """
    Returns:
      quant_exact: set of exact module prefixes like 'model.layers.0.self_attn.q_proj'
      quant_anc  : set of ancestor prefixes that contain quantized descendants
      by_layer   : dict layer_id -> set of module paths like 'self_attn.q_proj'
      info       : dict with counts and module_types
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {index_path}")

    wm = json.load(open(index_path, "r", encoding="utf-8"))["weight_map"]
    keys = list(wm.keys())

    dq = [k for k in keys if k.endswith(".dqweight")]
    qw = [k for k in keys if k.endswith(".qweight")]

    use = dq if dq else qw
    kind = "dqweight" if dq else ("qweight" if qw else "none")

    pat = re.compile(r"^model\.layers\.(\d+)\.(.+)\.(dqweight|qweight)$")

    quant_exact: Set[str] = set()
    quant_anc: Set[str] = set()
    by_layer: Dict[int, Set[str]] = {}
    module_types: Set[str] = set()

    for k in use:
        m = pat.match(k)
        if not m:
            continue
        layer = int(m.group(1))
        mod_path = m.group(2)  # e.g. self_attn.q_proj

        full_prefix = f"model.layers.{layer}.{mod_path}"
        quant_exact.add(full_prefix)

        by_layer.setdefault(layer, set()).add(mod_path)
        module_types.add(mod_path)

        # Add ancestors: model.layers.<layer>.<...> parents
        parts = full_prefix.split(".")
        # progressively add: model, model.layers, model.layers.<i>, model.layers.<i>.self_attn, ...
        for j in range(1, len(parts)):
            quant_anc.add(".".join(parts[:j]))

    info = {
        "kind": kind,
        "dqweight": len(dq),
        "qweight": len(qw),
        "module_types": sorted(module_types),
        "num_layers_in_index": len(by_layer),
    }
    return quant_exact, quant_anc, by_layer, info


# -----------------------------
# Model skeleton
# -----------------------------
def build_skeleton_from_config(model_dir: str):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return model


# -----------------------------
# Tree dumping
# -----------------------------
@dataclass
class DumpOptions:
    max_depth: int
    show_shapes: bool
    only_interesting: bool  # only show linears/norms/embeddings/attention/mlp blocks
    include_params_count: bool


def classify_interesting(name: str, module: torch.nn.Module) -> bool:
    """
    Heuristic filter when only_interesting is True.
    """
    cls = module.__class__.__name__.lower()
    n = name.lower()

    keywords = [
        "embed", "embedding",
        "norm", "rmsnorm",
        "attn", "attention",
        "mlp", "ffn",
        "proj", "linear",
        "lm_head",
        "rotary", "rope",
    ]
    if any(k in cls for k in keywords):
        return True
    if any(k in n for k in keywords):
        return True
    # Also keep modules that have a weight parameter (often key compute parts)
    if hasattr(module, "weight") and torch.is_tensor(getattr(module, "weight")):
        return True
    return False


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def module_weight_shape(module: torch.nn.Module) -> Optional[Tuple[int, ...]]:
    w = getattr(module, "weight", None)
    if torch.is_tensor(w):
        return tuple(w.shape)
    return None


def qmark(full_name: str, quant_exact: Set[str], quant_anc: Set[str]) -> str:
    if full_name in quant_exact:
        return "[Q] "
    if full_name in quant_anc:
        return "[Q+] "
    return "[FP]"


def dump_tree(
    module: torch.nn.Module,
    full_name: str,
    depth: int,
    lines: list,
    quant_exact: Set[str],
    quant_anc: Set[str],
    opt: DumpOptions,
):
    if depth > opt.max_depth:
        return

    # Decide whether to print this node
    do_print = True
    if opt.only_interesting and depth > 0:
        do_print = classify_interesting(full_name, module)

    if do_print:
        indent = "  " * depth
        mark = qmark(full_name, quant_exact, quant_anc)
        cls = module.__class__.__name__

        extras = []
        if opt.show_shapes:
            ws = module_weight_shape(module)
            if ws is not None:
                extras.append(f"weight={ws}")
        if opt.include_params_count:
            extras.append(f"params={count_params(module)}")

        extra_str = (" | " + ", ".join(extras)) if extras else ""
        lines.append(f"{indent}{mark}{full_name or '<root>'}: {cls}{extra_str}")

    # Recurse into children
    for child_name, child in module.named_children():
        child_full = child_name if full_name == "" else f"{full_name}.{child_name}"
        dump_tree(child, child_full, depth + 1, lines, quant_exact, quant_anc, opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="导出模型目录（含 config.json 与 model.safetensors.index.json）")
    ap.add_argument("--out_txt", type=str, required=True, help="输出 txt 路径")
    ap.add_argument("--max_depth", type=int, default=6, help="最大深度（根为0）")
    ap.add_argument("--show_shapes", action="store_true", help="对含 weight 的模块显示 weight shape")
    ap.add_argument("--only_interesting", action="store_true",
                    help="只输出较关键模块（embedding/norm/attn/mlp/linear/proj等），树更短更可读")
    ap.add_argument("--include_params_count", action="store_true",
                    help="显示每个模块（非递归）的参数量（recurse=False）")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)

    # config summary
    cfg_path = os.path.join(args.model_dir, "config.json")
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    qc = cfg.get("quantization_config", {})

    # quant info from index
    quant_exact, quant_anc, by_layer, info = load_quant_prefixes_from_index(args.model_dir)

    # build skeleton
    model = build_skeleton_from_config(args.model_dir)

    # dump
    opt = DumpOptions(
        max_depth=args.max_depth,
        show_shapes=args.show_shapes,
        only_interesting=args.only_interesting,
        include_params_count=args.include_params_count,
    )

    lines = []
    lines.append("==== quantization_config ====")
    for k in sorted(qc.keys()):
        lines.append(f"{k}: {qc[k]}")
    lines.append("")

    lines.append("==== quantized keys summary ====")
    lines.append(f"kind={info['kind']} | dqweight={info['dqweight']} | qweight={info['qweight']}")
    lines.append(f"unique quantized module types ({len(info['module_types'])}): {info['module_types']}")
    lines.append("")

    # a compact per-layer summary at the top (useful for inspection)
    lines.append("==== per-layer quantized modules (from index) ====")
    for lid in sorted(by_layer.keys()):
        mods = ", ".join(sorted(by_layer[lid]))
        lines.append(f"layer {lid:02d}: {mods}")
    lines.append("")

    lines.append("==== module tree (skeleton from config, no checkpoint loaded) ====")
    dump_tree(model, "", 0, lines, quant_exact, quant_anc, opt)

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] wrote tree to: {args.out_txt}")
    print(f"[INFO] max_depth={args.max_depth}, only_interesting={args.only_interesting}, show_shapes={args.show_shapes}")


if __name__ == "__main__":
    main()

