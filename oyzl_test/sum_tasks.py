#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 你当前跑的 5 个任务（无 hellaswag）
TASKS = ["boolq", "arc_easy", "arc_challenge", "piqa", "winogrande"]

# 任务主指标规则（严格按你指定）
PRIMARY_METRIC = {
    "boolq": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}

# 三个“表格”（一张 BASE + 两张量化配置）
VARIANTS = [
    "BASE",
    "mxfp_gptq_w4a4_hadamard_h128_mse_default",
    "mxfp_mrgptq_w4a4_hadamard_h128_mse_activation",
]

# 表格列（竖列）对应你三类模型
MODEL_IDS = ["llama-2-7b-hf", "llama-3-8b-hf", "Qwen3-8B"]


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_result_json(out_path: Path) -> Optional[Path]:
    """
    lm_eval --output_path 给的是目录时，里面通常会有 results.json / *.json
    这里做一个健壮的“候选查找”。
    """
    if out_path.is_file() and out_path.suffix == ".json":
        return out_path

    if not out_path.exists():
        return None

    # 常见文件名优先
    candidates = ["results.json", "result.json", "eval_results.json"]
    for name in candidates:
        p = out_path / name
        if p.is_file():
            return p

    # 顶层任意 json
    top_jsons = sorted(out_path.glob("*.json"))
    if top_jsons:
        return top_jsons[0]

    # 递归找（限制深度：最多 3 层）
    for depth in range(1, 4):
        for p in out_path.glob("*/" * depth + "*.json"):
            if p.is_file():
                return p

    return None


def _extract_task_metrics(blob: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    """
    兼容不同 lm_eval 输出结构：
    - {"results": {"boolq": {...}}}
    - {"results": {...}} 但 task key 可能不在
    """
    if not isinstance(blob, dict):
        return None

    if "results" in blob and isinstance(blob["results"], dict):
        r = blob["results"]
        if task in r and isinstance(r[task], dict):
            return r[task]

    # 有些情况下结果可能直接就是 task dict（极少见）
    if task in blob and isinstance(blob[task], dict):
        return blob[task]

    return None


def _get_metric(task_metrics: Dict[str, Any], name: str) -> Optional[float]:
    """
    lm_eval 里有时会出现 "acc,none" / "acc_norm,none" 这样的 key。
    这里做兼容：
    - 先取精确 key
    - 再取 name + ",..." 的第一个数值
    """
    if not isinstance(task_metrics, dict):
        return None

    v = task_metrics.get(name)
    if isinstance(v, (int, float)):
        return float(v)

    # 兼容 acc,none / acc_norm,none 等
    for k, v in task_metrics.items():
        if isinstance(k, str) and (k == name or k.startswith(name + ",")):
            if isinstance(v, (int, float)):
                return float(v)

    return None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    # lm_eval 通常是 0~1 的 accuracy；这里显示为百分数
    return f"{x*100:.2f}"


def _cell_text(task: str, acc: Optional[float], acc_norm: Optional[float]) -> str:
    """
    单元格展示：有 acc/acc_norm 就都标出来
    - ARC/PIQA：主指标 acc_norm（建议同时报 acc）
    - BoolQ/WinoGrande：主指标 acc（若 acc_norm 有也补充）
    """
    parts: List[str] = []

    primary = PRIMARY_METRIC[task]
    if primary == "acc_norm":
        # 主：acc_norm，补：acc
        if acc_norm is not None:
            parts.append(f"acc_norm={_fmt_pct(acc_norm)}")
        if acc is not None:
            parts.append(f"acc={_fmt_pct(acc)}")
    else:
        # 主：acc，补：acc_norm
        if acc is not None:
            parts.append(f"acc={_fmt_pct(acc)}")
        if acc_norm is not None:
            parts.append(f"acc_norm={_fmt_pct(acc_norm)}")

    return " (" .join([parts[0], " / ".join(parts[1:])]) + ")" if len(parts) >= 2 else (parts[0] if parts else "—")


def _pick_for_avg(task: str, acc: Optional[float], acc_norm: Optional[float]) -> Optional[float]:
    """
    平均规则：严格按任务主指标
    """
    primary = PRIMARY_METRIC[task]
    return acc_norm if primary == "acc_norm" else acc


def load_all_metrics(results_root: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    """
    返回结构：
    data[variant][model_id][task] = {"acc": float|None, "acc_norm": float|None}
    """
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    for variant in VARIANTS:
        data[variant] = {}
        for mid in MODEL_IDS:
            data[variant][mid] = {}
            for task in TASKS:
                data[variant][mid][task] = {"acc": None, "acc_norm": None}

    for mid in MODEL_IDS:
        for variant in VARIANTS:
            for task in TASKS:
                # 你的目录结构：OUT_DIR/<model_id>/<variant>/<task>_full/
                out_dir = results_root / mid / variant / f"{task}_full"
                jpath = _find_result_json(out_dir)
                if jpath is None:
                    continue

                blob = _read_json(jpath)
                if blob is None:
                    continue

                tm = _extract_task_metrics(blob, task)
                if tm is None:
                    continue

                acc = _get_metric(tm, "acc")
                acc_norm = _get_metric(tm, "acc_norm")

                data[variant][mid][task]["acc"] = acc
                data[variant][mid][task]["acc_norm"] = acc_norm

    return data


def make_table_md(data_variant: Dict[str, Dict[str, Dict[str, Optional[float]]]], title: str) -> str:
    """
    data_variant[model_id][task] = {"acc":..., "acc_norm":...}
    """
    lines: List[str] = []
    lines.append(f"### {title}")
    lines.append("")
    # 表头
    header = ["Task"] + MODEL_IDS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    # 任务行
    for task in TASKS:
        row = [task]
        for mid in MODEL_IDS:
            m = data_variant[mid][task]
            cell = _cell_text(task, m.get("acc"), m.get("acc_norm"))
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    # 平均行：按任务主指标取值再平均
    avg_row = ["Avg (task-wise primary)"]
    for mid in MODEL_IDS:
        vals: List[float] = []
        for task in TASKS:
            m = data_variant[mid][task]
            v = _pick_for_avg(task, m.get("acc"), m.get("acc_norm"))
            if isinstance(v, (int, float)):
                vals.append(float(v))
        avg = (sum(vals) / len(vals)) if vals else None
        # 同时标注用了几个任务
        avg_row.append(f"{_fmt_pct(avg)} (n={len(vals)})" if avg is not None else "—")
    lines.append("| " + " | ".join(avg_row) + " |")

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="lm_eval 结果根目录（例如：/cephfs/.../lm_eval_results_tasks_llama_2_3-Qwen_3）",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="lm_eval_summary.md",
        help="输出 md 文件路径（默认：lm_eval_summary.md）",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    out_md = Path(args.out_md).resolve()

    data = load_all_metrics(results_root)

    md_lines: List[str] = []
    md_lines.append("# LM-Eval Summary (BoolQ / ARC / PIQA / WinoGrande)")
    md_lines.append("")
    md_lines.append("**显示格式**：单元格内尽量同时展示 `acc` 与 `acc_norm`（若存在）。")
    md_lines.append("")
    md_lines.append("**平均规则（严格按任务）**：")
    md_lines.append("- BoolQ：用 `acc`")
    md_lines.append("- ARC-Easy：用 `acc_norm`（若存在同时展示 `acc`；平均只取 `acc_norm`）")
    md_lines.append("- ARC-Challenge：用 `acc_norm`（同上）")
    md_lines.append("- PIQA：用 `acc_norm`（同上）")
    md_lines.append("- WinoGrande：用 `acc`")
    md_lines.append("")
    md_lines.append("> 数值以百分数显示（例如 73.25 表示 73.25%）。Avg 行括号内 n 表示参与平均的任务数。")
    md_lines.append("")

    # 三张表：BASE / 两个量化版本
    for variant in VARIANTS:
        title = "BASE" if variant == "BASE" else variant
        # 重排为 data_variant[mid][task]...
        data_variant: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        for mid in MODEL_IDS:
            data_variant[mid] = data[variant][mid]
        md_lines.append(make_table_md(data_variant, title))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()

