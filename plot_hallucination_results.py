import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "hallucination_unified_summary.csv"


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["hallucination_rate"] = float(row["hallucination_rate"])
            row["support_rate"] = float(row["support_rate"])
            row["hallucinated"] = int(row["hallucinated"])
            row["supported"] = int(row["supported"])
            row["unverifiable"] = int(row["unverifiable"])
            row["total"] = int(row["total"])
            rows.append(row)
    return rows


def build_matrix(rows, metric):
    conditions = sorted({r["condition"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    values = np.zeros((len(conditions), len(datasets)), dtype=float)
    for i, cond in enumerate(conditions):
        for j, ds in enumerate(datasets):
            match = next(r for r in rows if r["condition"] == cond and r["dataset"] == ds)
            values[i, j] = match[metric]
    return conditions, datasets, values


def plot_grouped_bar(conditions, datasets, values, title, ylabel, out_dir: Path, out_name, percent=False):
    x = np.arange(len(datasets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cond in enumerate(conditions):
        bars = ax.bar(x + (i - 1.5) * width, values[i], width, label=cond)
        for b in bars:
            v = b.get_height()
            label = f"{v*100:.1f}%" if percent else f"{int(v)}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.01 if percent else max(values.max() * 0.01, 1)),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    vmax = float(np.nanmax(values)) if np.size(values) else 0.0
    if percent:
        ax.set_ylim(0, min(1.0, vmax * 1.2 + 0.02) if vmax > 0 else 1.0)
    else:
        ax.set_ylim(0, vmax * 1.2 + 5 if vmax > 0 else 1.0)
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / out_name, dpi=200)
    plt.close(fig)


def plot_dataset_breakdown(rows, out_dir: Path):
    conditions = sorted({r["condition"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    metrics = ["hallucinated", "supported", "unverifiable"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]

    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(conditions))
        x = np.arange(len(conditions))
        for metric, color in zip(metrics, colors):
            vals = np.array([next(r for r in ds_rows if r["condition"] == c)[metric] for c in conditions])
            ax.bar(x, vals, bottom=bottom, label=metric, color=color)
            bottom += vals

        totals = np.array([next(r for r in ds_rows if r["condition"] == c)["total"] for c in conditions])
        for i, t in enumerate(totals):
            ax.text(i, t + 2, f"n={t}", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"{ds}: Verdict Count Breakdown by Condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        filename = f"dataset_breakdown_{ds.lower().replace('-', '_')}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Bar charts from hallucination_unified_summary.csv")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "plots",
        help="Directory for PNG outputs (default: ./plots)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Missing file: {SUMMARY_CSV}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(SUMMARY_CSV)

    conditions, datasets, h_values = build_matrix(rows, "hallucination_rate")
    _, _, s_values = build_matrix(rows, "support_rate")
    _, _, hc_values = build_matrix(rows, "hallucinated")
    _, _, sc_values = build_matrix(rows, "supported")
    _, _, uc_values = build_matrix(rows, "unverifiable")

    plot_grouped_bar(
        conditions,
        datasets,
        h_values,
        "Hallucination Rate by Dataset and Condition",
        "Hallucination rate",
        out_dir,
        "hallucination_rate_grouped.png",
        percent=True,
    )
    plot_grouped_bar(
        conditions,
        datasets,
        s_values,
        "Support Rate by Dataset and Condition",
        "Support rate",
        out_dir,
        "support_rate_grouped.png",
        percent=True,
    )
    plot_grouped_bar(
        conditions,
        datasets,
        hc_values,
        "Hallucinated Count by Dataset and Condition",
        "Hallucinated count",
        out_dir,
        "hallucinated_count_grouped.png",
        percent=False,
    )
    plot_grouped_bar(
        conditions,
        datasets,
        sc_values,
        "Supported Count by Dataset and Condition",
        "Supported count",
        out_dir,
        "supported_count_grouped.png",
        percent=False,
    )
    plot_grouped_bar(
        conditions,
        datasets,
        uc_values,
        "Unverifiable Count by Dataset and Condition",
        "Unverifiable count",
        out_dir,
        "unverifiable_count_grouped.png",
        percent=False,
    )
    plot_dataset_breakdown(rows, out_dir)

    print(f"Saved charts to: {out_dir}")


if __name__ == "__main__":
    main()
