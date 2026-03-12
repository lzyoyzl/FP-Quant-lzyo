#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List


TASKS = ["winogrande", "hellaswag"]  # 你现在已完成的两类日志
# 如果后续也要加 gsm8k_llama / mmlu_cot_llama，可以在这里扩展


def parse_model_name(name: str) -> Dict[str, str]:
    """
    从模型名里解析出分类字段，用于“分类总结”。
    例如：nvfp_gptq_w4a4_hadamard_h128_minmax_activation
    """
    if name == "BASE":
        return {
            "model": name,
            "fmt": "BASE",
            "algo": "BASE",
            "rotate": "",
            "h": "",
            "observer": "",
            "order": "",
        }

    tokens = name.split("_")
    fmt = tokens[0] if tokens else ""
    algo = tokens[1] if len(tokens) > 1 else ""

    rotate = "identity"
    h = ""
    if "hadamard" in tokens:
        rotate = "hadamard"
        # 找 h16/h32/h128 这类
        for t in tokens:
            if re.fullmatch(r"h\d+", t):
                h = t[1:]
                break

    observer = ""
    if "minmax" in tokens:
        observer = "minmax"
    elif "mse" in tokens:
        observer = "mse"
    else:
        # RTN 通常默认就是 minmax（你的导出脚本里也是这么用的）
        if algo == "rtn":
            observer = "minmax"

    order = ""
    if tokens and tokens[-1] in ("default", "activation"):
        order = tokens[-1]
    else:
        # 没写出来时，一般默认就是 default
        if algo in ("rtn", "gptq", "mrgptq"):
            order = "default"

    return {
        "model": name,
        "fmt": fmt,
        "algo": algo,
        "rotate": rotate,
        "h": h,
        "observer": observer,
        "order": order,
    }


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _is_separator_row(line: str) -> bool:
    """
    判断是否为表格分隔线，例如：
    |---------|------:|------|-----:|--------|---|-----:|---|-----:|
    """
    body = line.replace("|", "").strip()
    if not body:
        return False
    return all(c in "-: " for c in body)


def parse_lmeval_table_lines(text: str) -> List[Dict[str, str]]:
    """
    解析 lm_eval 输出的表格行（形如 |task|...|metric|...|value|...|stderr|）。
    返回所有匹配行（包含 task/metric/value/stderr）。

    关键修复：
    1) 支持“同一 task 的第二行指标”在 Tasks 列留空的情况（acc_norm 行）：
       - 若 task 为空，则使用上一条非空 task 进行 forward-fill。
    2) 更稳健跳过表头分隔线（|---------|... 这种）。
    3) 不要求行必须以 | 结尾（有些日志可能没末尾 |）。
    """
    rows: List[Dict[str, str]] = []
    last_task = ""

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if _is_separator_row(line):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 10:
            continue

        task = parts[1]
        metric = parts[5]
        value = parts[7]
        stderr = parts[9]

        # 跳过表头行
        if task.lower() in ("tasks", "task") and metric.lower() == "metric":
            continue

        # forward-fill：Tasks 列为空则沿用上一行 task（这就是 acc_norm 行的情况）
        if task == "":
            task = last_task
        else:
            last_task = task

        if task in TASKS and metric:
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "stderr": stderr,
                }
            )

    return rows


def choose_metric(task: str, metric_values: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    给定一个 task 的所有 metric -> (value, stderr)，选择“准确率”对应的那一个。
    """
    if task == "hellaswag":
        # hellaswag 只取 acc_norm（不回退 acc）
        if "acc_norm" in metric_values and metric_values["acc_norm"][0] is not None:
            v, s = metric_values["acc_norm"]
            return "acc_norm", v, s
        return None, None, None

    if task == "winogrande":
        if "acc" in metric_values and metric_values["acc"][0] is not None:
            v, s = metric_values["acc"]
            return "acc", v, s
        # 兜底：如果只有别的 acc*，选第一个可用
        for m, (v, s) in metric_values.items():
            if v is not None and m.startswith("acc"):
                return m, v, s
        return None, None, None

    return None, None, None


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    col_w = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_row(r):
        return "| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(headers))) + " |"

    sep = "| " + " | ".join("-" * col_w[i] for i in range(len(headers))) + " |"

    out = [fmt_row(headers), sep]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="lm_eval_logs_3tasks", help="日志目录（包含 *__.log）")
    ap.add_argument("--out_md", type=str, default="lm_eval_summary.md", help="输出 Markdown 文件")
    ap.add_argument("--out_csv", type=str, default="lm_eval_summary.csv", help="输出 CSV 文件")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"[ERROR] log_dir not found: {log_dir}")

    # model -> task -> metric -> (value, stderr)
    store: Dict[str, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]] = {}

    log_files = sorted(log_dir.glob("*.log"))
    for lf in log_files:
        if lf.name.startswith("master_"):
            continue
        stem = lf.stem
        if "__" not in stem:
            continue

        model, task = stem.split("__", 1)
        if task not in TASKS:
            continue

        text = lf.read_text(errors="ignore")
        rows = parse_lmeval_table_lines(text)

        # 用“最后出现的那一行”覆盖（适合日志里多次打印同一任务的情况）
        store.setdefault(model, {}).setdefault(task, {})
        for r in rows:
            if r["task"] != task:
                continue
            metric = r["metric"]
            v = safe_float(r["value"])
            s = safe_float(r["stderr"])
            store[model][task][metric] = (v, s)

    # 汇总：每个模型按任务挑一个“准确率”指标
    summary_rows = []
    for model in sorted(store.keys(), key=lambda x: (x != "BASE", x)):
        info = parse_model_name(model)

        row = {
            **info,
            "winogrande_metric": "",
            "winogrande": "",
            "winogrande_stderr": "",
            "hellaswag_metric": "",
            "hellaswag": "",
            "hellaswag_stderr": "",
            "avg_2tasks": "",
        }

        vals_for_avg = []

        for task in TASKS:
            metrics = store.get(model, {}).get(task, {})
            m, v, s = choose_metric(task, metrics)
            if v is not None:
                vals_for_avg.append(v)
            if task == "winogrande":
                row["winogrande_metric"] = m or ""
                row["winogrande"] = f"{v:.4f}" if v is not None else ""
                row["winogrande_stderr"] = f"{s:.4f}" if s is not None else ""
            elif task == "hellaswag":
                row["hellaswag_metric"] = m or ""
                row["hellaswag"] = f"{v:.4f}" if v is not None else ""
                row["hellaswag_stderr"] = f"{s:.4f}" if s is not None else ""

        if len(vals_for_avg) == 2:
            row["avg_2tasks"] = f"{(vals_for_avg[0] + vals_for_avg[1]) / 2:.4f}"

        summary_rows.append(row)

    # 分组输出：BASE / mxfp / nvfp / other
    groups = {"BASE": [], "mxfp": [], "nvfp": [], "other": []}
    for r in summary_rows:
        fmt = r["fmt"]
        if fmt == "BASE":
            groups["BASE"].append(r)
        elif fmt == "mxfp":
            groups["mxfp"].append(r)
        elif fmt == "nvfp":
            groups["nvfp"].append(r)
        else:
            groups["other"].append(r)

    headers = [
        "model", "algo", "rotate", "h", "observer", "order",
        "winogrande(acc)", "hellaswag(acc_norm)", "avg(2 tasks)"
    ]

    md_parts = []
    for gname in ["BASE", "mxfp", "nvfp", "other"]:
        if not groups[gname]:
            continue
        md_parts.append(f"## {gname}\n")
        rows = []
        for r in groups[gname]:
            rows.append([
                r["model"],
                r["algo"],
                r["rotate"],
                r["h"],
                r["observer"],
                r["order"],
                r["winogrande"],
                r["hellaswag"],
                r["avg_2tasks"],
            ])
        md_parts.append(markdown_table(headers, rows))
        md_parts.append("")

    out_md = "\n".join(md_parts).strip() + "\n"
    Path(args.out_md).write_text(out_md, encoding="utf-8")

    # CSV
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "fmt", "algo", "rotate", "h", "observer", "order",
            "winogrande_metric", "winogrande", "winogrande_stderr",
            "hellaswag_metric", "hellaswag", "hellaswag_stderr",
            "avg_2tasks",
        ])
        for r in summary_rows:
            w.writerow([
                r["model"], r["fmt"], r["algo"], r["rotate"], r["h"], r["observer"], r["order"],
                r["winogrande_metric"], r["winogrande"], r["winogrande_stderr"],
                r["hellaswag_metric"], r["hellaswag"], r["hellaswag_stderr"],
                r["avg_2tasks"],
            ])

    print(f"[OK] Wrote:\n  - {args.out_md}\n  - {args.out_csv}\n")
    print(out_md)


if __name__ == "__main__":
    main()

