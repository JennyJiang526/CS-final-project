"""
evaluate.py
-----------
Evaluates accuracy and hallucination metrics across four conditions:
  - baseline (no RAG)
  - cot (Chain of Thought)
  - meta (Meta prompting)
  - medrag (MedRAG adaptive-k)

Accuracy metrics (where ground truth allows exact comparison):
  - MedQA-US : letter match (A/B/C/D/E)
  - PubMedQA : yes/no/maybe match

Hallucination proxy metrics (all datasets):
  - Unsupported Sentence Ratio  : fraction of sentences in answer not
                                   supported by ground truth (BERTScore < threshold)
  - Refusal Rate                : fraction of answers that are empty / refusals

For medrag only:
  - Citation Coverage           : fraction of explanation sentences that
                                   have at least one citation listed

Usage:
    python evaluate.py

Outputs:
    evaluation_results.json   - full per-condition, per-dataset breakdown
    evaluation_summary.csv    - printable summary table
"""

import json
import re
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── optional: bert_score for unsupported sentence ratio ──────────────────────
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("[WARNING] bert_score not installed. Unsupported Sentence Ratio will be skipped.")
    print("          Install with: pip install bert-score")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FILES = {
    "baseline": ROOT / "results_v2" / "baseline_results_aligned.jsonl",
    "cot":      ROOT / "results_v2" / "cot_results_aligned.jsonl",
    "meta":     ROOT / "results_v2" / "meta_results_aligned.jsonl",
    "medrag":   ROOT / "medrag_results_clean.jsonl",
    "medrag_fixed_k": ROOT / "medrag_results_fixed_k.jsonl",
}

BERT_THRESHOLD = 0.85   # F1 threshold to consider a sentence "supported"
REFUSAL_PHRASES = [
    "i don't know", "i do not know", "i cannot", "i can't",
    "i'm not sure", "i am not sure", "unable to answer",
    "no information", "not enough information", "cannot determine",
    "as an ai", "i'm sorry", "i apologize",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(Path(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_LETTER_KEYPHRASES = (
    "answer is",
    "answer:",
    "therefore",
    "conclusion",
    "final answer",
)


def extract_letter(text):
    """
    Extract MedQA-style option letter (A–E). Avoid matching letters in prose
    (e.g. 'A critical factor') by preferring text after cue phrases, then tail.
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    tl = t.lower()

    best_kw_idx = -1
    best_letter = None
    for phrase in _LETTER_KEYPHRASES:
        start = 0
        plen = len(phrase)
        while True:
            idx = tl.find(phrase, start)
            if idx == -1:
                break
            after = t[idx + plen :]
            m = re.search(r"\b([A-Ea-e])\b", after)
            if m and idx > best_kw_idx:
                best_kw_idx = idx
                best_letter = m.group(1).upper()
            start = idx + 1

    if best_letter is not None:
        return best_letter

    tail = t[-200:] if len(t) > 200 else t
    m = re.search(r"\b([A-Ea-e])\b", tail)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-Ea-e])\b", t)
    return m.group(1).upper() if m else None


def extract_pubmed_decision(text):
    """Extract yes/no/maybe from a string."""
    if not text:
        return None
    t = text.strip().lower()
    for label in ["yes", "no", "maybe"]:
        if t.startswith(label):
            return label
    m = re.search(r'\b(yes|no|maybe)\b', t)
    return m.group(1) if m else None


def extract_gt_pubmed(gt_text):
    """Extract final_decision from PubMedQA ground truth."""
    m = re.search(r'<final_decision>\s*(\w+)', gt_text)
    return m.group(1).lower() if m else None


def is_refusal(text):
    if not text:
        return True
    t = text.lower()
    return any(phrase in t for phrase in REFUSAL_PHRASES)


def split_sentences(text):
    """Naive sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def get_answer(record, condition):
    """Unified answer extraction across conditions."""
    if condition in ("medrag", "medrag_fixed_k"):
        return record.get("llm_answer", "") or ""
    else:
        return record.get("model_answer", "") or ""


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(records, condition):
    """
    Returns dict: {dataset: {"correct": int, "total": int, "accuracy": float}}
    Only computed for MedQA-US and PubMedQA.
    """
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in records:
        ds = rec["dataset"]
        gt = rec.get("ground_truth", "")
        answer = get_answer(rec, condition)

        if ds == "MedQA-US":
            pred = extract_letter(answer)
            truth = extract_letter(gt)
            if truth is not None:
                results[ds]["total"] += 1
                if pred == truth:
                    results[ds]["correct"] += 1

        elif ds == "PubMedQA":
            pred = extract_pubmed_decision(answer)
            truth = extract_gt_pubmed(gt)
            if truth is not None:
                results[ds]["total"] += 1
                if pred == truth:
                    results[ds]["correct"] += 1

    for ds in results:
        t = results[ds]["total"]
        c = results[ds]["correct"]
        results[ds]["accuracy"] = round(c / t, 4) if t > 0 else 0.0

    return dict(results)


# ─────────────────────────────────────────────────────────────────────────────
# REFUSAL RATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_refusal_rate(records, condition):
    """Returns dict: {dataset: {"refusals": int, "total": int, "refusal_rate": float}}"""
    results = defaultdict(lambda: {"refusals": 0, "total": 0})

    for rec in records:
        ds = rec["dataset"]
        answer = get_answer(rec, condition)
        results[ds]["total"] += 1
        if is_refusal(answer):
            results[ds]["refusals"] += 1

    for ds in results:
        t = results[ds]["total"]
        r = results[ds]["refusals"]
        results[ds]["refusal_rate"] = round(r / t, 4) if t > 0 else 0.0

    return dict(results)


# ─────────────────────────────────────────────────────────────────────────────
# CITATION COVERAGE (medrag only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_citation_coverage(records):
    """
    For medrag: fraction of explanation sentences that have >= 1 citation.
    Returns dict: {dataset: {"covered": int, "total_sentences": int, "coverage": float}}
    """
    results = defaultdict(lambda: {"covered": 0, "total_sentences": 0})

    for rec in records:
        ds = rec["dataset"]
        explanation = rec.get("llm_explanation", [])
        if not isinstance(explanation, list):
            continue
        for item in explanation:
            results[ds]["total_sentences"] += 1
            citations = item.get("citations", [])
            if citations:
                results[ds]["covered"] += 1

    for ds in results:
        t = results[ds]["total_sentences"]
        c = results[ds]["covered"]
        results[ds]["coverage"] = round(c / t, 4) if t > 0 else 0.0

    return dict(results)


# ─────────────────────────────────────────────────────────────────────────────
# UNSUPPORTED SENTENCE RATIO  (requires bert_score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_unsupported_ratio(records, condition, sample_size=100):
    """
    For each answer, split into sentences. For each sentence, compute
    BERTScore F1 against the ground truth. Sentences below threshold
    are counted as unsupported.

    To keep cost manageable, sample up to `sample_size` records per dataset.
    Returns dict: {dataset: {"unsupported": int, "total": int, "ratio": float}}
    """
    if not BERT_AVAILABLE:
        return {}

    from bert_score import score as bert_score_fn

    # group by dataset
    by_dataset = defaultdict(list)
    for rec in records:
        by_dataset[rec["dataset"]].append(rec)

    results = {}

    for ds, recs in by_dataset.items():
        sample = recs[:sample_size]
        total_sentences = 0
        unsupported = 0

        all_sentences = []
        all_references = []

        for rec in sample:
            answer = get_answer(rec, condition)
            gt = rec.get("ground_truth", "")
            sentences = split_sentences(answer)
            for sent in sentences:
                all_sentences.append(sent)
                all_references.append(gt[:1000])  # truncate very long GTs

        if not all_sentences:
            continue

        print(f"  Computing BERTScore for {ds} ({len(all_sentences)} sentences)...")
        _, _, F1 = bert_score_fn(
            all_sentences, all_references,
            lang="en", verbose=False
        )

        for score in F1.tolist():
            total_sentences += 1
            if score < BERT_THRESHOLD:
                unsupported += 1

        results[ds] = {
            "unsupported": unsupported,
            "total_sentences": total_sentences,
            "unsupported_ratio": round(unsupported / total_sentences, 4) if total_sentences > 0 else 0.0,
            "note": f"sampled {len(sample)} records, threshold={BERT_THRESHOLD}"
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for condition, path in FILES.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {condition}  ({path})")
        print(f"{'='*50}")

        path = Path(path)
        if not path.is_file():
            print(f"  [SKIP] File not found: {path}")
            continue

        records = load_jsonl(path)
        print(f"  Loaded {len(records)} records")

        result = {}

        # Accuracy
        print("  Computing accuracy...")
        result["accuracy"] = compute_accuracy(records, condition)

        # Refusal rate
        print("  Computing refusal rate...")
        result["refusal_rate"] = compute_refusal_rate(records, condition)

        # Citation coverage (MedRAG-style outputs only)
        if condition in ("medrag", "medrag_fixed_k"):
            print("  Computing citation coverage...")
            result["citation_coverage"] = compute_citation_coverage(records)

        # Unsupported sentence ratio
        if BERT_AVAILABLE:
            print("  Computing unsupported sentence ratio (BERTScore)...")
            result["unsupported_ratio"] = compute_unsupported_ratio(records, condition)
        else:
            result["unsupported_ratio"] = "bert_score not available"

        all_results[condition] = result

    # ── Save full JSON ────────────────────────────────────────────────────────
    out_json = ROOT / "evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_json}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ACCURACY SUMMARY")
    print("="*70)
    datasets_acc = ["MedQA-US", "PubMedQA"]
    header = f"{'Condition':<12}" + "".join(f"{ds:>18}" for ds in datasets_acc)
    print(header)
    print("-"*70)
    for cond in FILES:
        if cond not in all_results:
            continue
        acc = all_results[cond].get("accuracy", {})
        row = f"{cond:<12}"
        for ds in datasets_acc:
            if ds in acc:
                val = f"{acc[ds]['accuracy']*100:.1f}% ({acc[ds]['correct']}/{acc[ds]['total']})"
            else:
                val = "N/A"
            row += f"{val:>18}"
        print(row)

    print("\n" + "="*70)
    print("REFUSAL RATE SUMMARY")
    print("="*70)
    datasets_all = ["MedQA-US", "PubMedQA", "BioASQ", "MediQA"]
    header = f"{'Condition':<12}" + "".join(f"{ds:>14}" for ds in datasets_all)
    print(header)
    print("-"*70)
    for cond in FILES:
        if cond not in all_results:
            continue
        rr = all_results[cond].get("refusal_rate", {})
        row = f"{cond:<12}"
        for ds in datasets_all:
            if ds in rr:
                val = f"{rr[ds]['refusal_rate']*100:.1f}%"
            else:
                val = "N/A"
            row += f"{val:>14}"
        print(row)

    print("\n" + "="*70)
    print("CITATION COVERAGE (MedRAG / MedRAG fixed-k)")
    print("="*70)
    for cc_cond in ("medrag", "medrag_fixed_k"):
        if cc_cond not in all_results:
            continue
        cc = all_results[cc_cond].get("citation_coverage")
        if not cc:
            continue
        print(f"  --- {cc_cond} ---")
        for ds, vals in cc.items():
            print(f"  {ds}: {vals['coverage']*100:.1f}% ({vals['covered']}/{vals['total_sentences']} sentences cited)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_rows = []
    for cond in FILES:
        if cond not in all_results:
            continue
        for ds in datasets_all:
            row = {"condition": cond, "dataset": ds}
            acc = all_results[cond].get("accuracy", {}).get(ds, {})
            row["accuracy"] = acc.get("accuracy", "") if acc else ""
            row["correct"] = acc.get("correct", "") if acc else ""
            row["total_acc"] = acc.get("total", "") if acc else ""
            rr = all_results[cond].get("refusal_rate", {}).get(ds, {})
            row["refusal_rate"] = rr.get("refusal_rate", "") if rr else ""
            row["refusals"] = rr.get("refusals", "") if rr else ""
            row["total_rr"] = rr.get("total", "") if rr else ""
            if cond in ("medrag", "medrag_fixed_k"):
                cc = all_results[cond].get("citation_coverage", {}).get(ds, {})
                row["citation_coverage"] = cc.get("coverage", "") if cc else ""
            else:
                row["citation_coverage"] = ""
            csv_rows.append(row)

    out_csv = ROOT / "evaluation_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV summary saved to {out_csv}")


if __name__ == "__main__":
    main()
