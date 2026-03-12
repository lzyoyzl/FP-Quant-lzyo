#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List


TASKS = ["winogrande", "hellaswag", "gsm8k_llama"]  # 已完成三类日志


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
        if algo == "rtn":
            observer = "minmax"

    order = ""
    if tokens and tokens[-1] in ("default", "activation"):
        order = tokens[-1]
    else:
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
    body = line.replace("|", "").strip()
    if not body:
        return False
    return all(c in "-: " for c in body)


def parse_lmeval_table_lines(text: str) -> List[Dict[str, str]]:
    """
    解析 lm_eval 输出表格。
    返回匹配行：task/filter/metric/value/stderr

    支持：
    - Tasks 列留空（同一 task 多行指标）forward-fill
    - 跳过分隔线
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
        # ["", "gsm8k_llama", "3", "flexible_extract", "8", "exact_match", "↑", "0.7324", "±", "0.0122", ""]
        if len(parts) < 10:
            continue

        task = parts[1]
        filt = parts[3]   # Filter 列
        metric = parts[5]
        value = parts[7]
        stderr = parts[9]

        # 跳过表头
        if task.lower() in ("tasks", "task") and metric.lower() == "metric":
            continue

        if task == "":
            task = last_task
        else:
            last_task = task

        if task in TASKS and metric:
            rows.append(
                {
                    "task": task,
                    "filter": filt,
                    "metric": metric,
                    "value": value,
                    "stderr": stderr,
                }
            )
    return rows


def choose_metric(task: str, metric_values: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    给定一个 task 的所有 metric_key -> (value, stderr)，选择用于汇总的指标。
    metric_key 规则：
    - winogrande/hellaswag: key 就是 metric 名，如 "acc_norm"
    - gsm8k_llama: key 是 "filter::metric"，如 "flexible_extract::exact_match"
    """
    if task == "hellaswag":
        # 只取 acc_norm（不回退 acc）
        if metric_values.get("acc_norm", (None, None))[0] is not None:
            v, s = metric_values["acc_norm"]
            return "acc_norm", v, s
        return None, None, None

    if task == "winogrande":
        if metric_values.get("acc", (None, None))[0] is not None:
            v, s = metric_values["acc"]
            return "acc", v, s
        for m, (v, s) in metric_values.items():
            if v is not None and m.startswith("acc"):
                return m, v, s
        return None, None, None

    if task == "gsm8k_llama":
        # 优先 flexible_extract 的 exact_match（与你日志示例一致，也是常用主指标）
        k1 = "flexible_extract::exact_match"
        if metric_values.get(k1, (None, None))[0] is not None:
            v, s = metric_values[k1]
            return k1, v, s

        # 其次 strict_match 的 exact_match
        k2 = "strict_match::exact_match"
        if metric_values.get(k2, (None, None))[0] is not None:
            v, s = metric_values[k2]
            return k2, v, s

        # 再兜底：任意 *::exact_match
        for k, (v, s) in metric_values.items():
            if v is not None and k.endswith("::exact_match"):
                return k, v, s

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

    # model -> task -> metric_key -> (value, stderr)
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

        store.setdefault(model, {}).setdefault(task, {})

        for r in rows:
            if r["task"] != task:
                continue

            metric = r["metric"]
            filt = r["filter"] or ""

            # gsm8k 需要把 filter 编进 key，避免 flexible/strict 覆盖
            if task == "gsm8k_llama":
                key = f"{filt}::{metric}" if filt else metric
            else:
                key = metric

            v = safe_float(r["value"])
            s = safe_float(r["stderr"])
            store[model][task][key] = (v, s)

    # 汇总
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
            "gsm8k_metric": "",
            "gsm8k": "",
            "gsm8k_stderr": "",
            "avg_3tasks": "",
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
            elif task == "gsm8k_llama":
                row["gsm8k_metric"] = m or ""
                row["gsm8k"] = f"{v:.4f}" if v is not None else ""
                row["gsm8k_stderr"] = f"{s:.4f}" if s is not None else ""

        if len(vals_for_avg) == 3:
            row["avg_3tasks"] = f"{(vals_for_avg[0] + vals_for_avg[1] + vals_for_avg[2]) / 3:.4f}"

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
        "winogrande(acc)",
        "hellaswag(acc_norm)",
        "gsm8k(exact_match@flexible)",
        "avg(3 tasks)"
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
                r["gsm8k"],
                r["avg_3tasks"],
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
            "gsm8k_metric", "gsm8k", "gsm8k_stderr",
            "avg_3tasks",
        ])
        for r in summary_rows:
            w.writerow([
                r["model"], r["fmt"], r["algo"], r["rotate"], r["h"], r["observer"], r["order"],
                r["winogrande_metric"], r["winogrande"], r["winogrande_stderr"],
                r["hellaswag_metric"], r["hellaswag"], r["hellaswag_stderr"],
                r["gsm8k_metric"], r["gsm8k"], r["gsm8k_stderr"],
                r["avg_3tasks"],
            ])

    print(f"[OK] Wrote:\n  - {args.out_md}\n  - {args.out_csv}\n")
    print(out_md)


if __name__ == "__main__":
    main()

