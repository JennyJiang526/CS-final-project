"""
按 medrag_results_clean.jsonl 的顺序，用 (dataset, question) 从 baseline/cot/meta
结果中各取首次匹配行，写出 *_aligned.jsonl（行数与顺序与 medrag 一致）。
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEDRAG_PATH = ROOT / "medrag_results_clean.jsonl"
SOURCES = {
    "baseline_results_aligned.jsonl": ROOT / "results_v2" / "baseline_results.jsonl",
    "cot_results_aligned.jsonl": ROOT / "results_v2" / "cot_results.jsonl",
    "meta_results_aligned.jsonl": ROOT / "results_v2" / "meta_results.jsonl",
}
OUT_DIR = ROOT / "results_v2"


def first_occurrence_index(path: Path) -> dict[tuple[str, str], dict]:
    idx: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            key = (o["dataset"], o["question"])
            if key not in idx:
                idx[key] = o
    return idx


def align_one(
    medrag_rows: list[dict],
    index: dict[tuple[str, str], dict],
) -> tuple[list[dict], int]:
    out: list[dict] = []
    unmatched = 0
    for m in medrag_rows:
        key = (m["dataset"], m["question"])
        hit = index.get(key)
        if hit is None:
            unmatched += 1
            out.append(
                {
                    "_unmatched": True,
                    "dataset": m["dataset"],
                    "question": m["question"],
                }
            )
        else:
            out.append(hit)
    return out, unmatched


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MEDRAG_PATH, encoding="utf-8") as f:
        medrag_rows = [json.loads(line) for line in f if line.strip()]

    n_med = len(medrag_rows)
    print(f"medrag_results_clean.jsonl 基准行数: {n_med}\n")

    for out_name, src_path in SOURCES.items():
        if not src_path.is_file():
            raise FileNotFoundError(f"缺少源文件: {src_path}")

        index = first_occurrence_index(src_path)
        aligned, n_miss = align_one(medrag_rows, index)
        assert len(aligned) == n_med

        out_path = OUT_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in aligned:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        by_ds = Counter(r["dataset"] for r in aligned)
        print(f"=== {out_name} ===")
        print(f"  总行数: {len(aligned)}")
        print("  各 dataset 行数:")
        for ds in sorted(by_ds.keys()):
            print(f"    {ds}: {by_ds[ds]}")
        print(f"  未匹配到的 question 数: {n_miss}")
        print(f"  已写入: {out_path}\n")


if __name__ == "__main__":
    main()
