"""
evaluate_bertscore.py
---------------------
BioASQ 与 MediQA：对 baseline / cot / meta / medrag 计算 BERTScore F1
（candidate vs ground_truth 前 1000 字符），汇总统计并写入 bertscore_results.json。
"""

import json
import statistics
from pathlib import Path

from bert_score import score as bert_score_fn

ROOT = Path(__file__).resolve().parent

FILES = {
    "baseline": ROOT / "results_v2" / "baseline_results_aligned.jsonl",
    "cot": ROOT / "results_v2" / "cot_results_aligned.jsonl",
    "meta": ROOT / "results_v2" / "meta_results_aligned.jsonl",
    "medrag": ROOT / "medrag_results_clean.jsonl",
    "medrag_fixed_k": ROOT / "medrag_results_fixed_k.jsonl",
}

TARGET_DATASETS = ("BioASQ", "MediQA")
REF_MAX_LEN = 1000


def get_candidate(record: dict, condition: str) -> str:
    if condition in ("medrag", "medrag_fixed_k"):
        return (record.get("llm_answer") or "").strip()
    return (record.get("model_answer") or "").strip()


def get_reference(record: dict) -> str:
    gt = record.get("ground_truth") or ""
    return gt[:REF_MAX_LEN]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f1_stats(f1_list: list[float]) -> dict:
    if not f1_list:
        return {
            "n": 0,
            "mean_f1": None,
            "max_f1": None,
            "min_f1": None,
            "std_f1": None,
        }
    n = len(f1_list)
    mean_v = statistics.mean(f1_list)
    max_v = max(f1_list)
    min_v = min(f1_list)
    if n < 2:
        std_v = 0.0
    else:
        std_v = statistics.stdev(f1_list)
    return {
        "n": n,
        "mean_f1": round(mean_v, 6),
        "max_f1": round(max_v, 6),
        "min_f1": round(min_v, 6),
        "std_f1": round(std_v, 6),
    }


def run_bertscore_for_pairs(candidates: list[str], references: list[str]) -> list[float]:
    _, _, f1 = bert_score_fn(candidates, references, lang="en", verbose=False)
    return [float(x) for x in f1.tolist()]


def main() -> None:
    results: dict = {}

    for condition, path in FILES.items():
        if not path.is_file():
            print(f"[SKIP] {condition}: file not found {path}")
            continue

        records = load_jsonl(path)
        by_ds: dict[str, list[tuple[str, str]]] = {ds: [] for ds in TARGET_DATASETS}

        for rec in records:
            ds = rec.get("dataset")
            if ds not in TARGET_DATASETS:
                continue
            cand = get_candidate(rec, condition)
            ref = get_reference(rec)
            by_ds[ds].append((cand, ref))

        results[condition] = {}
        for ds in TARGET_DATASETS:
            pairs = by_ds[ds]
            candidates = [p[0] for p in pairs]
            references = [p[1] for p in pairs]

            if not candidates:
                results[condition][ds] = f1_stats([])
                continue

            f1_list = run_bertscore_for_pairs(candidates, references)
            results[condition][ds] = f1_stats(f1_list)

    out_path = ROOT / "bertscore_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 打印汇总表 ──────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("BERTScore F1 (candidate vs ground_truth[:1000]), lang=en")
    print("=" * 88)

    for ds in TARGET_DATASETS:
        print(f"\n--- {ds} ---")
        header = f"{'Condition':<10} {'n':>6} {'mean_F1':>10} {'max_F1':>10} {'min_F1':>10} {'std_F1':>10}"
        print(header)
        print("-" * len(header))
        for cond in sorted(results.keys()):
            s = results[cond][ds]
            if s["n"] == 0:
                row = f"{cond:<10} {0:>6} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            else:
                row = (
                    f"{cond:<10} {s['n']:>6} "
                    f"{s['mean_f1']:>10.4f} {s['max_f1']:>10.4f} "
                    f"{s['min_f1']:>10.4f} {s['std_f1']:>10.4f}"
                )
            print(row)

    print(f"\n已保存: {out_path}\n")


if __name__ == "__main__":
    main()
