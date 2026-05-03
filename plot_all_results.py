"""
plot_all_results.py
-------------------
Orchestrates bar charts under plots/:
  plots/hallucination/  — from hallucination_unified_summary.csv (via plot_hallucination_results.py)
  plots/evaluation/     — from evaluation_summary.csv
  plots/bertscore/      — from bertscore_results.json
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
SUB = {
    "hallucination": PLOTS / "hallucination",
    "evaluation": PLOTS / "evaluation",
    "bertscore": PLOTS / "bertscore",
}


def _safe_float(x) -> float:
    if x is None or str(x).strip() == "":
        return float("nan")
    return float(x)


def _load_eval_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _plot_eval_grouped(rows, metric_key: str, title: str, ylabel: str, datasets: list[str], out_path: Path, percent: bool):
    conditions = sorted({r["condition"] for r in rows})
    x = np.arange(len(datasets))
    width = min(0.8 / max(len(conditions), 1), 0.18)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cond in enumerate(conditions):
        vals = []
        for ds in datasets:
            r = next((row for row in rows if row["condition"] == cond and row["dataset"] == ds), None)
            v = _safe_float(r[metric_key]) if r else float("nan")
            vals.append(v)
        vals = np.array(vals, dtype=float)
        offset = (i - (len(conditions) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=cond)
        for b in bars:
            v = b.get_height()
            if np.isnan(v):
                continue
            lab = f"{v*100:.1f}%" if percent else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, lab, ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.25)
    subvals = np.array(
        [_safe_float(r[metric_key]) for r in rows if r["dataset"] in datasets],
        dtype=float,
    )
    ymax = float(np.nanmax(subvals)) if subvals.size else 0.0
    if percent:
        ax.set_ylim(0, min(1.05, ymax * 1.15 + 0.02) if ymax > 0 else 1.0)
    else:
        ax.set_ylim(0, ymax * 1.15 + 0.02 if ymax > 0 else 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_evaluation(out_dir: Path) -> None:
    path = ROOT / "evaluation_summary.csv"
    if not path.is_file():
        print(f"[skip] evaluation: missing {path.name}")
        return
    rows = _load_eval_rows(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_eval_grouped(
        rows,
        "accuracy",
        "Accuracy (MedQA-US & PubMedQA)",
        "Accuracy",
        ["MedQA-US", "PubMedQA"],
        out_dir / "accuracy_grouped.png",
        percent=False,
    )
    _plot_eval_grouped(
        rows,
        "refusal_rate",
        "Refusal rate (all datasets)",
        "Refusal rate",
        ["MedQA-US", "PubMedQA", "BioASQ", "MediQA"],
        out_dir / "refusal_rate_grouped.png",
        percent=True,
    )

    # Citation coverage: medrag vs medrag_fixed_k only
    cc_rows = [r for r in rows if str(r.get("citation_coverage", "")).strip() != ""]
    if cc_rows:
        conds = sorted({r["condition"] for r in cc_rows})
        dss = ["MedQA-US", "PubMedQA", "BioASQ", "MediQA"]
        x = np.arange(len(dss))
        width = min(0.35, 0.8 / max(len(conds), 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cond in enumerate(conds):
            vals = []
            for ds in dss:
                m = next((r for r in cc_rows if r["condition"] == cond and r["dataset"] == ds), None)
                vals.append(_safe_float(m["citation_coverage"]) if m else float("nan"))
            vals = np.array(vals, dtype=float)
            offset = (i - (len(conds) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=cond)
            for b in bars:
                v = b.get_height()
                if np.isnan(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_title("Citation coverage (MedRAG outputs)")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Coverage")
        ax.set_xticks(x)
        ax.set_xticklabels(dss)
        ax.legend(title="Condition")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "citation_coverage_grouped.png", dpi=200)
        plt.close(fig)


def plot_bertscore(out_dir: Path) -> None:
    path = ROOT / "bertscore_results.json"
    if not path.is_file():
        print(f"[skip] bertscore: missing {path.name}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in ("BioASQ", "MediQA"):
        conds = sorted(data.keys())
        x = np.arange(len(conds))
        means = [float(data[c][ds]["mean_f1"]) if data[c].get(ds) and data[c][ds].get("mean_f1") is not None else float("nan") for c in conds]
        means = np.array(means, dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, color="#4c72b0")
        for b, m in zip(bars, means):
            if np.isnan(m):
                continue
            ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"BERTScore mean F1 — {ds}")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Mean F1")
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=20, ha="right")
        mx = float(np.nanmax(means)) if means.size else 0.0
        ax.set_ylim(0, min(1.05, mx * 1.12 + 0.02) if mx > 0 else 1.0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"bertscore_mean_f1_{ds.lower()}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    SUB["hallucination"].mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_hallucination_results.py"),
            "--out-dir",
            str(SUB["hallucination"]),
        ],
        check=True,
        cwd=str(ROOT),
    )
    plot_evaluation(SUB["evaluation"])
    plot_bertscore(SUB["bertscore"])
    print("Plots root:", PLOTS.resolve())
    for name, p in SUB.items():
        print(f"  {name}/ -> {p.resolve()}")


if __name__ == "__main__":
    main()
