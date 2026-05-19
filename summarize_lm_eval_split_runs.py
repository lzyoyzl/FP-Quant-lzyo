#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TASKS = ["boolq", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]
TASK_METRIC = {
    "boolq": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "hellaswag": "acc_norm",
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


def parse_lmeval_table_rows(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    last_task = ""

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if _is_separator_row(line):
            continue

        parts = [p.strip() for p in line.split("|")]
        # typical row:
        # ["", "boolq", "1", "none", "0", "acc", "↑", "0.85", "±", "0.01", ""]
        if len(parts) < 10:
            continue

        task = parts[1]
        filt = parts[3]
        metric = parts[5]
        value = parts[7]
        stderr = parts[9]

        if task.lower() in ("tasks", "task", "groups", "group"):
            continue
        if metric.lower() == "metric":
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


def parse_log_name(log_name: str) -> Optional[Tuple[str, str, str]]:
    stem = Path(log_name).stem

    # BASE__task.log
    m_base = re.match(r"^BASE__(boolq|arc_easy|arc_challenge|piqa|winogrande|hellaswag)$", stem)
    if m_base:
        return "BASE", "BASE", m_base.group(1)

    # ROTSEARCH__model__task.log
    m_rot = re.match(
        r"^ROTSEARCH__(.+)__(boolq|arc_easy|arc_challenge|piqa|winogrande|hellaswag)$", stem
    )
    if m_rot:
        return "ROTSEARCH", m_rot.group(1), m_rot.group(2)

    # LEGACY__model__task.log
    m_legacy = re.match(
        r"^LEGACY__(.+)__(boolq|arc_easy|arc_challenge|piqa|winogrande|hellaswag)$", stem
    )
    if m_legacy:
        return "LEGACY", m_legacy.group(1), m_legacy.group(2)

    return None


def fmt(v: Optional[float], ndigits: int = 4) -> str:
    return "" if v is None else f"{v:.{ndigits}f}"


def fmt_pct(v: Optional[float], ndigits: int = 2) -> str:
    return "" if v is None else f"{v:.{ndigits}f}%"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def row_str(r: List[str]) -> str:
        return "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    out = [row_str(headers), sep]
    out.extend(row_str(r) for r in rows)
    return "\n".join(out)


def build_model_row(
    source: str,
    model: str,
    task_to_value: Dict[str, Optional[float]],
    base_avg: Optional[float],
) -> Dict[str, Optional[float] | str]:
    vals = [task_to_value.get(t) for t in TASKS]
    valid = [v for v in vals if v is not None]
    avg = sum(valid) / len(valid) if len(valid) == len(TASKS) else None

    if model == "BASE":
        recovery = 100.0
    elif avg is not None and base_avg not in (None, 0):
        recovery = avg / base_avg * 100.0
    else:
        recovery = None

    return {
        "source": source,
        "model": model,
        "boolq": task_to_value.get("boolq"),
        "arc_easy": task_to_value.get("arc_easy"),
        "arc_challenge": task_to_value.get("arc_challenge"),
        "piqa": task_to_value.get("piqa"),
        "winogrande": task_to_value.get("winogrande"),
        "hellaswag": task_to_value.get("hellaswag"),
        "avg": avg,
        "recovery": recovery,
    }


def model_fmt(model_name: str) -> str:
    if model_name.startswith("nvfp"):
        return "nvfp"
    if model_name.startswith("mxfp"):
        return "mxfp"
    if model_name == "BASE":
        return "BASE"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_dirs",
        nargs="+",
        default=["lm_eval_logs_llama3_rotsearch_only", "lm_eval_logs_llama3_legacy_only"],
        help="One or more lm_eval log directories.",
    )
    ap.add_argument(
        "--out_md",
        default="lm_eval_summary_llama3_split.md",
        help="Output markdown path.",
    )
    args = ap.parse_args()

    log_dirs = [Path(p) for p in args.log_dirs if Path(p).is_dir()]
    if not log_dirs:
        raise SystemExit("[ERROR] No valid log_dirs found.")

    # store[source][model][task] = value
    store: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {"BASE": {}, "ROTSEARCH": {}, "LEGACY": {}}
    for d in log_dirs:
        for lf in sorted(d.glob("*.log")):
            if lf.name.startswith("master_"):
                continue

            parsed_name = parse_log_name(lf.name)
            if parsed_name is None:
                continue
            source, model, task_from_name = parsed_name

            rows = parse_lmeval_table_rows(lf.read_text(errors="ignore"))
            value_for_task: Optional[float] = None
            target_metric = TASK_METRIC[task_from_name]

            for r in rows:
                if r["task"] != task_from_name:
                    continue
                if r["metric"] == target_metric:
                    value_for_task = safe_float(r["value"])
                    # first exact match is enough
                    break

            store.setdefault(source, {}).setdefault(model, {})[task_from_name] = value_for_task

    # BASE row
    base_task_vals = store.get("BASE", {}).get("BASE", {})
    base_row = build_model_row("BASE", "BASE", base_task_vals, base_avg=None)
    base_avg = base_row["avg"] if isinstance(base_row["avg"], float) else None
    # Set base recovery to exactly 100 when base avg exists; still show 100 by requirement.
    base_row["recovery"] = 100.0

    # Build all model rows
    rows_all: List[Dict[str, Optional[float] | str]] = [base_row]
    for source in ("ROTSEARCH", "LEGACY"):
        for model in sorted(store.get(source, {}).keys()):
            rows_all.append(
                build_model_row(
                    source=source,
                    model=model,
                    task_to_value=store[source][model],
                    base_avg=base_avg,
                )
            )

    # Section split
    nvfp_rows = [r for r in rows_all if r["model"] != "BASE" and model_fmt(str(r["model"])) == "nvfp"]
    mxfp_rows = [r for r in rows_all if r["model"] != "BASE" and model_fmt(str(r["model"])) == "mxfp"]

    # stable ordering: source first then model
    nvfp_rows.sort(key=lambda r: (str(r["source"]), str(r["model"])))
    mxfp_rows.sort(key=lambda r: (str(r["source"]), str(r["model"])))

    headers = [
        "source",
        "model",
        "boolq(acc)",
        "arc_easy(acc_norm)",
        "arc_challenge(acc_norm)",
        "piqa(acc_norm)",
        "winogrande(acc)",
        "hellaswag(acc_norm)",
        "avg",
        "Recovery",
    ]

    base_table_rows = [
        [
            "BASE",
            "BASE",
            fmt(base_row["boolq"]),
            fmt(base_row["arc_easy"]),
            fmt(base_row["arc_challenge"]),
            fmt(base_row["piqa"]),
            fmt(base_row["winogrande"]),
            fmt(base_row["hellaswag"]),
            fmt(base_row["avg"]),
            fmt_pct(base_row["recovery"]),
        ]
    ]

    def to_table_rows(items: List[Dict[str, Optional[float] | str]]) -> List[List[str]]:
        out: List[List[str]] = []
        for r in items:
            out.append(
                [
                    str(r["source"]),
                    str(r["model"]),
                    fmt(r["boolq"]),
                    fmt(r["arc_easy"]),
                    fmt(r["arc_challenge"]),
                    fmt(r["piqa"]),
                    fmt(r["winogrande"]),
                    fmt(r["hellaswag"]),
                    fmt(r["avg"]),
                    fmt_pct(r["recovery"]),
                ]
            )
        return out

    md_parts: List[str] = []
    md_parts.append("# Llama-3-8B-Instruct Eval Summary")
    md_parts.append("")
    md_parts.append("## BASE")
    md_parts.append("")
    md_parts.append(markdown_table(headers, base_table_rows))
    md_parts.append("")
    md_parts.append("## NVFP (ROTSEARCH + LEGACY)")
    md_parts.append("")
    md_parts.append(markdown_table(headers, to_table_rows(nvfp_rows)) if nvfp_rows else "_No NVFP rows found._")
    md_parts.append("")
    md_parts.append("## MXFP (ROTSEARCH + LEGACY)")
    md_parts.append("")
    md_parts.append(markdown_table(headers, to_table_rows(mxfp_rows)) if mxfp_rows else "_No MXFP rows found._")
    md_parts.append("")

    out_md = Path(args.out_md)
    out_md.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"[OK] Wrote markdown summary: {out_md}")


if __name__ == "__main__":
    main()
