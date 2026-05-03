"""
evaluate_hallucination_unified.py
----------------------------------
Unified hallucination evaluation for four settings (baseline / cot / meta / medrag).

Uses one judge prompt: question + model answer + ground truth -> Claude decides whether
the answer contains factually incorrect claims. All datasets are scored for fair comparison.

Verdicts:
  SUPPORTED     - Consistent with ground truth; no incorrect factual claims
  HALLUCINATED  - At least one claim contradicts or is not supported by ground truth
  UNVERIFIABLE  - Ground truth too vague to decide

Outputs:
  hallucination_unified_results.json  - per-record verdicts (by condition key)
  hallucination_unified_summary.csv   - aggregated table

Incremental run (e.g. add medrag_fixed_k without re-judging baseline/cot/meta/medrag):
  python evaluate_hallucination_unified.py --only medrag_fixed_k
  Merges into the existing JSON/CSV if present; otherwise creates them.

Re-judge rows with verdict ERROR only (full Q/answer from source JSONL by index), then rewrite JSON + CSV:
  python evaluate_hallucination_unified.py --retry-errors
"""

import argparse
import json
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent


def log(*args, **kwargs):
    """Print with flush so progress shows immediately in terminals / logs."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.is_file():
        return
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v:
            os.environ[k] = v


_load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Judge model; if you get 404 / model not found, set CLAUDE_JUDGE_MODEL in .env to a valid model id.
MODEL = os.environ.get("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-5")
# One-token auth smoke test only (not the judge model). 401 here means the API key is invalid.
_AUTH_SMOKE_MODEL = "claude-haiku-4-5-20251001"

FILES = {
    "baseline": ROOT / "results_v2" / "baseline_results_aligned.jsonl",
    "cot":      ROOT / "results_v2" / "cot_results_aligned.jsonl",
    "meta":     ROOT / "results_v2" / "meta_results_aligned.jsonl",
    "medrag":   ROOT / "medrag_results_clean.jsonl",
    # MedRAG fixed-k: same schema as medrag; answer from llm_answer
    "medrag_fixed_k": ROOT / "medrag_results_fixed_k.jsonl",
}

SLEEP_BETWEEN = 0.5   # seconds between API calls
GT_MAX_CHARS  = 1500  # truncate long ground truths
ANS_MAX_CHARS = 1000  # truncate long answers
# Log progress every N records (also logs the first and last record).
PROGRESS_EVERY = 10

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are a medical fact-checker evaluating whether a model's answer contains hallucinations.

Question: {question}

Ground truth reference:
{ground_truth}

Model's answer:
{model_answer}

Your task: Determine whether the model's answer contains any factually incorrect or unsupported statements when compared to the ground truth reference.

Rules:
- Focus only on factual claims, not on writing style or completeness
- Minor paraphrasing of correct facts is acceptable (SUPPORTED)
- If the model says something that directly contradicts the ground truth, that is a hallucination (HALLUCINATED)
- If the model adds specific facts not mentioned in the ground truth and they cannot be verified, lean towards HALLUCINATED
- If the ground truth is too short or vague to make a determination, use UNVERIFIABLE

Reply with ONLY one of these three words:
SUPPORTED
HALLUCINATED
UNVERIFIABLE"""

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


def get_answer(record, condition):
    if condition in ("medrag", "medrag_fixed_k"):
        return record.get("llm_answer", "") or ""
    return record.get("model_answer", "") or ""


def truncate(text, max_chars):
    if not text:
        return ""
    text = str(text)
    return text[:max_chars] + "..." if len(text) > max_chars else text


def call_judge(client, question, ground_truth, model_answer, retries=3):
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=truncate(ground_truth, GT_MAX_CHARS),
        model_answer=truncate(model_answer, ANS_MAX_CHARS),
    )

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            verdict = message.content[0].text.strip().upper()
            for valid in ["SUPPORTED", "HALLUCINATED", "UNVERIFIABLE"]:
                if valid in verdict:
                    return valid
            return verdict
        except Exception as e:
            log(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(records, condition, client):
    results = []
    total = len(records)
    log(f"  [{condition}] Starting: {total} records")

    for i, rec in enumerate(records):
        ds = rec["dataset"]
        question = rec.get("question", "")
        ground_truth = rec.get("ground_truth", "")
        model_answer = get_answer(rec, condition)

        verdict = call_judge(client, question, ground_truth, model_answer)

        results.append({
            "dataset": ds,
            "question": question[:100],
            "model_answer": model_answer[:200],
            "verdict": verdict,
        })

        time.sleep(SLEEP_BETWEEN)

        n = i + 1
        if n == 1 or n == total or n % PROGRESS_EVERY == 0:
            log(f"  [{condition}] {n}/{total} dataset={ds} verdict={verdict}")

    log(f"  [{condition}] Finished {total}/{total}")
    return results


def retry_errors_in_results(client, all_results_out: dict) -> int:
    """
    For each verdict == ERROR, reload the aligned record from FILES[condition] by index
    and call the judge again. Updates all_results_out in place.
    """
    retried = 0
    for condition, verdicts in list(all_results_out.items()):
        if not isinstance(verdicts, list):
            continue
        if condition not in FILES:
            log(f"[retry-errors] skip unknown condition key: {condition!r}")
            continue
        path = Path(FILES[condition])
        if not path.is_file():
            log(f"[retry-errors] skip {condition}: file not found {path}")
            continue
        records = load_jsonl(path)
        if len(records) != len(verdicts):
            log(
                f"[retry-errors] WARN {condition}: JSONL rows={len(records)} "
                f"!= saved verdicts={len(verdicts)}; retry uses min index bound"
            )
        err_idx = [i for i, v in enumerate(verdicts) if v.get("verdict") == "ERROR"]
        if not err_idx:
            continue
        log(f"[retry-errors] {condition}: {len(err_idx)} ERROR row(s) to re-judge")
        for i in err_idx:
            if i >= len(records):
                log(f"  [{condition}] index {i}: ERROR but no JSONL row — skipped")
                continue
            rec = records[i]
            ds = rec.get("dataset", "")
            question = rec.get("question", "")
            ground_truth = rec.get("ground_truth", "")
            model_answer = get_answer(rec, condition)
            verdict = call_judge(client, question, ground_truth, model_answer)
            verdicts[i]["dataset"] = ds
            verdicts[i]["question"] = question[:100]
            verdicts[i]["model_answer"] = model_answer[:200]
            verdicts[i]["verdict"] = verdict
            retried += 1
            time.sleep(SLEEP_BETWEEN)
            log(f"  [{condition}] re-judged index={i} dataset={ds} verdict={verdict}")
    return retried


def aggregate(results):
    by_dataset = defaultdict(lambda: defaultdict(int))
    for rec in results:
        by_dataset[rec["dataset"]][rec["verdict"]] += 1

    summary = {}
    for ds, counts in by_dataset.items():
        total = sum(v for k, v in counts.items() if k != "ERROR")
        hallucinated  = counts.get("HALLUCINATED", 0)
        supported     = counts.get("SUPPORTED", 0)
        unverifiable  = counts.get("UNVERIFIABLE", 0)
        errors        = counts.get("ERROR", 0)
        summary[ds] = {
            "total": total,
            "supported": supported,
            "hallucinated": hallucinated,
            "unverifiable": unverifiable,
            "errors": errors,
            "hallucination_rate": round(hallucinated / total, 4) if total > 0 else 0,
            "support_rate": round(supported / total, 4) if total > 0 else 0,
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Unified hallucination judge over result JSONL files.")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--only",
        metavar="CONDITION",
        help="Run only this condition key (e.g. medrag_fixed_k). Merges into existing hallucination_unified_results.json and hallucination_unified_summary.csv.",
    )
    g.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-call judge only for verdict ERROR (uses source JSONL by index), then rewrite JSON + CSV.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    only = (args.only or "").strip() or None
    if only and only not in FILES:
        log(f"ERROR: unknown --only {only!r}. Valid: {', '.join(FILES)}")
        return

    # Reload .env and read the key at runtime (do not rely only on import-time env).
    _load_dotenv()
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        log("ERROR: ANTHROPIC_API_KEY is missing. Add to project .env: ANTHROPIC_API_KEY=sk-ant-...")
        return

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    client = anthropic.Anthropic(api_key=api_key)

    # Smoke test: 401 means bad key (not a MODEL issue). If this passes but later calls fail, check MODEL / billing.
    try:
        client.messages.create(
            model=_AUTH_SMOKE_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        log(f"OK: Anthropic auth passed (smoke model={_AUTH_SMOKE_MODEL}). Judge model={MODEL}")
    except anthropic.AuthenticationError as e:
        log("ERROR: Anthropic AuthenticationError (401 invalid x-api-key).")
        log("  Fix: create a new API key in Anthropic Console, paste the FULL key on ONE line in .env, no quotes.")
        log(f"  Detail: {e}")
        return
    except anthropic.PermissionDeniedError as e:
        log("ERROR: Anthropic PermissionDeniedError (billing / org access).")
        log(f"  Detail: {e}")
        return

    if args.retry_errors:
        out_json = ROOT / "hallucination_unified_results.json"
        if not out_json.is_file():
            log(f"ERROR: missing {out_json.name}; nothing to retry.")
            return
        with open(out_json, encoding="utf-8") as f:
            all_results_out = json.load(f)
        if not isinstance(all_results_out, dict):
            log("ERROR: results JSON must be an object keyed by condition.")
            return
        n = retry_errors_in_results(client, all_results_out)
        log(f"\n[retry-errors] Re-judged {n} row(s).")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results_out, f, indent=2, ensure_ascii=False)
        log(f"Full results saved to {out_json}")
        all_summaries_out = {c: aggregate(v) for c, v in all_results_out.items()}
        datasets = ["MedQA-US", "PubMedQA", "BioASQ", "MediQA"]
        log("\n" + "=" * 75)
        log("HALLUCINATION RATE SUMMARY (% HALLUCINATED) after retry")
        log("=" * 75)
        header = f"{'Condition':<16}" + "".join(f"{ds:>16}" for ds in datasets)
        log(header)
        log("-" * 75)
        csv_rows = []
        for cond in sorted(all_summaries_out.keys()):
            row_str = f"{cond:<16}"
            for ds in datasets:
                vals = all_summaries_out[cond].get(ds, {})
                if vals:
                    val = f"{vals['hallucination_rate']*100:.1f}% ({vals['total']})"
                else:
                    val = "N/A"
                row_str += f"{val:>16}"
            log(row_str)
            for ds in datasets:
                vals = all_summaries_out[cond].get(ds, {})
                csv_rows.append({
                    "condition": cond,
                    "dataset": ds,
                    "hallucination_rate": vals.get("hallucination_rate", ""),
                    "support_rate": vals.get("support_rate", ""),
                    "hallucinated": vals.get("hallucinated", ""),
                    "supported": vals.get("supported", ""),
                    "unverifiable": vals.get("unverifiable", ""),
                    "total": vals.get("total", ""),
                })
        out_csv = ROOT / "hallucination_unified_summary.csv"
        fieldnames = [
            "condition", "dataset", "hallucination_rate", "support_rate",
            "hallucinated", "supported", "unverifiable", "total",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        log(f"\nCSV saved to {out_csv}")
        return

    all_results   = {}
    all_summaries = {}

    files_iter = [(only, FILES[only])] if only else list(FILES.items())
    for condition, path in files_iter:
        log(f"\n{'='*55}")
        log(f"Evaluating: {condition}")
        log(f"{'='*55}")

        path = Path(path)
        if not path.is_file():
            log(f"  [SKIP] File not found: {path}")
            continue

        records = load_jsonl(path)
        log(f"  Loaded {len(records)} records")
        log(f"  Estimated API calls: {len(records)}")

        verdicts = evaluate_condition(records, condition, client)
        summary  = aggregate(verdicts)

        all_results[condition]   = verdicts
        all_summaries[condition] = summary

        # print interim summary
        log(f"\n  {condition} summary:")
        for ds, vals in summary.items():
            log(f"    {ds}: hallucination={vals['hallucination_rate']*100:.1f}% "
                f"supported={vals['support_rate']*100:.1f}% "
                f"(n={vals['total']})")

    if not all_results:
        log("ERROR: nothing evaluated (missing result file(s) or empty run).")
        return

    # ── Merge + save full results ─────────────────────────────────────────────
    out_json = ROOT / "hallucination_unified_results.json"
    if only and out_json.is_file():
        with open(out_json, encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, dict):
            merged = {}
        merged.update(all_results)
        all_results_out = merged
        log(f"\nMerged {list(all_results.keys())} into existing {out_json.name}")
    else:
        all_results_out = all_results

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results_out, f, indent=2, ensure_ascii=False)
    log(f"\nFull results saved to {out_json}")

    # One summary dict per condition (from merged JSON so CSV/table stay consistent)
    all_summaries_out = {c: aggregate(v) for c, v in all_results_out.items()}

    # ── Print final summary table ─────────────────────────────────────────────
    datasets = ["MedQA-US", "PubMedQA", "BioASQ", "MediQA"]
    log("\n" + "="*75)
    log("HALLUCINATION RATE SUMMARY (% HALLUCINATED)")
    log("="*75)
    header = f"{'Condition':<16}" + "".join(f"{ds:>16}" for ds in datasets)
    log(header)
    log("-"*75)

    cond_order = sorted(all_summaries_out.keys())
    csv_rows = []
    for cond in cond_order:
        if cond not in all_summaries_out:
            continue
        row_str = f"{cond:<16}"
        for ds in datasets:
            vals = all_summaries_out[cond].get(ds, {})
            if vals:
                val = f"{vals['hallucination_rate']*100:.1f}% ({vals['total']})"
            else:
                val = "N/A"
            row_str += f"{val:>16}"
        log(row_str)

        for ds in datasets:
            vals = all_summaries_out[cond].get(ds, {})
            csv_rows.append({
                "condition": cond,
                "dataset": ds,
                "hallucination_rate": vals.get("hallucination_rate", ""),
                "support_rate": vals.get("support_rate", ""),
                "hallucinated": vals.get("hallucinated", ""),
                "supported": vals.get("supported", ""),
                "unverifiable": vals.get("unverifiable", ""),
                "total": vals.get("total", ""),
            })

    # ── Save CSV (full table; same keys as merged JSON)
    out_csv = ROOT / "hallucination_unified_summary.csv"
    fieldnames = ["condition", "dataset", "hallucination_rate", "support_rate", "hallucinated", "supported", "unverifiable", "total"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log(f"\nCSV saved to {out_csv}")


if __name__ == "__main__":
    main()
