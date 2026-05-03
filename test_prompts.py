"""
Citation-Aware Prompt Test Script
Datasets with context: BioASQ, PubMedQA (10 each)
MedQA-US and MediQA skipped — no context field available yet (pending RAG)
"""

import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI


def _load_dotenv() -> None:
    """Load OPENAI_API_KEY from .env next to this script (not committed)."""
    p = Path(__file__).resolve().parent / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("OPENAI_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val:
                os.environ.setdefault("OPENAI_API_KEY", val)
            break


_load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_PATH = "medical_qa_1000.jsonl"
OUTPUT_DIR = "results_v2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prompt_test_results.jsonl")
SAMPLES_PER_DATASET = 10
RANDOM_SEED = 42
MODEL = "gpt-4o"

# Datasets with no context are skipped until RAG is available
SKIP_DATASETS = {"MedQA-US", "MediQA"}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Prompt Templates ─────────────────────────────────────────────────────────
PROMPT_YES_NO = """\
You are a medical research assistant. You will be given a biomedical \
yes/no/maybe question and a list of retrieved reference snippets.

Your task:
1. Give a final decision: answer must be exactly one of: "yes", "no", or "maybe".
2. Write an explanation (2-5 sentences) supporting your decision.
   - Every sentence in the explanation MUST cite at least one snippet \
using [n] notation, where n is the snippet number.
   - Only make factual claims that are directly supported by the provided snippets.
   - If the evidence is mixed or insufficient, your decision should be "maybe" \
and your explanation must reflect the uncertainty.
   - Do NOT introduce medical knowledge not present in the snippets.

Retrieved Snippets:
{snippets}

Question:
{question}

Respond ONLY with a JSON object in this exact format. No preamble, no markdown:
{{
  "answer": "<yes|no|maybe>",
  "explanation": [
    {{"sentence": "<sentence text>", "citations": [<list of snippet indices>]}},
    ...
  ]
}}"""

PROMPT_FACTOID = """\
You are a biomedical question answering assistant. You will be given a \
factoid question and a list of retrieved reference snippets.

Your task:
1. Provide a concise direct answer (a word, phrase, gene name, drug name, etc.).
2. Write a supporting explanation (1-3 sentences) that justifies your answer.
   - Every sentence in the explanation MUST cite at least one snippet \
using [n] notation, where n is the snippet number.
   - Only make factual claims that are directly supported by the provided snippets.
   - Do NOT introduce medical knowledge not present in the snippets.

Retrieved Snippets:
{snippets}

Question:
{question}

Respond ONLY with a JSON object in this exact format. No preamble, no markdown:
{{
  "answer": "<concise direct answer>",
  "explanation": [
    {{"sentence": "<sentence text>", "citations": [<list of snippet indices>]}},
    ...
  ]
}}"""

PROMPT_MAP = {
    "PubMedQA": PROMPT_YES_NO,
    "BioASQ":   PROMPT_FACTOID,
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def format_snippets(context: str) -> tuple[str, list[str]]:
    """
    Treat the whole context string as a single snippet for now.
    Returns (formatted string for prompt, list of snippet strings for eval).
    When RAG is plugged in, replace this with a list of retrieved chunks.
    """
    snippets = [context.strip()]
    formatted = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    return formatted, snippets


def call_gpt4o(prompt: str) -> tuple[str, dict | None]:
    """Call GPT-4o and return (raw_text, parsed_json_or_None)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if GPT-4o wraps output anyway
    clean = raw
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        parsed = None
        print(f"    [WARN] JSON parse failed: {e}")

    return raw, parsed


# ── Evaluation helpers ────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as rouge_lib
    _rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("[WARN] rouge_score not installed — skipping lexical overlap metric")
    print("       Run: pip install rouge-score")

OVERLAP_THRESHOLD = 0.15  # calibrate on manual sample before reporting


def evaluate_response(parsed: dict, snippets: list[str]) -> dict:
    if parsed is None:
        return {"citation_coverage": None, "unsupported_ratio": None,
                "avg_lexical_overlap": None, "parse_error": True}

    explanation = parsed.get("explanation", [])
    total = len(explanation)
    if total == 0:
        return {"citation_coverage": 0.0, "unsupported_ratio": 1.0,
                "avg_lexical_overlap": 0.0, "parse_error": False}

    supported = 0
    overlap_scores = []

    for item in explanation:
        sentence   = item.get("sentence", "")
        cited_idxs = item.get("citations", [])
        has_citation = len(cited_idxs) > 0

        max_overlap = 0.0
        if ROUGE_AVAILABLE:
            for idx in cited_idxs:
                if 1 <= idx <= len(snippets):
                    snippet = snippets[idx - 1]
                    score = _rouge.score(snippet, sentence)
                    max_overlap = max(max_overlap, score["rougeL"].recall)

        is_supported = has_citation and (
            max_overlap >= OVERLAP_THRESHOLD if ROUGE_AVAILABLE else True
        )
        if is_supported:
            supported += 1
        overlap_scores.append(max_overlap)

    return {
        "citation_coverage":   round(supported / total, 4),
        "unsupported_ratio":   round(1 - supported / total, 4),
        "avg_lexical_overlap": round(sum(overlap_scores) / len(overlap_scores), 4),
        "parse_error": False,
    }


# ── Load and sample ───────────────────────────────────────────────────────────
def load_samples(path: str, n: int, seed: int) -> dict[str, list[dict]]:
    from collections import defaultdict
    buckets: dict[str, list[dict]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r["dataset"]
            if d in SKIP_DATASETS:
                continue
            if r.get("context", "").strip():
                buckets[d].append(r)

    rng = random.Random(seed)
    return {d: rng.sample(records, min(n, len(records)))
            for d, records in buckets.items()}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_samples(DATASET_PATH, SAMPLES_PER_DATASET, RANDOM_SEED)
    # 只重跑 PubMedQA
    samples = {d: recs for d, recs in samples.items() if d == "PubMedQA"}

    print(f"Datasets to run: {list(samples.keys())}")
    print(f"Skipped (no context): {SKIP_DATASETS}")
    print(f"Output: {OUTPUT_PATH}\n")

    results = []

    for dataset, records in samples.items():
        template = PROMPT_MAP[dataset]
        print(f"── {dataset} ({len(records)} samples) ──────────────────────")

        for i, record in enumerate(records):
            if record["dataset"] != "PubMedQA":
                continue
            formatted_snippets, snippet_list = format_snippets(record["context"])
            prompt = template.format(
                snippets=formatted_snippets,
                question=record["question"],
            )

            print(f"  [{i+1}/{len(records)}] calling GPT-4o...", end=" ", flush=True)
            raw, parsed = call_gpt4o(prompt)
            metrics = evaluate_response(parsed, snippet_list)
            print(f"coverage={metrics['citation_coverage']}  "
                  f"unsupported={metrics['unsupported_ratio']}  "
                  f"overlap={metrics['avg_lexical_overlap']}")

            result = {
                "dataset":        dataset,
                "question":       record["question"],
                "ground_truth":   record["answer"],
                "llm_answer":     parsed.get("answer") if parsed else None,
                "llm_explanation": parsed.get("explanation") if parsed else None,
                "raw_output":     raw,
                "snippets":       snippet_list,
                "metrics":        metrics,
            }
            results.append(result)

            # Be gentle with the API
            time.sleep(0.5)

        print()

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print("═" * 55)
    print("SUMMARY")
    print("═" * 55)
    for dataset in samples:
        subset = [r for r in results if r["dataset"] == dataset]
        valid  = [r for r in subset if not r["metrics"]["parse_error"]]
        if not valid:
            print(f"{dataset}: all parse errors")
            continue
        avg_cov     = sum(r["metrics"]["citation_coverage"] for r in valid) / len(valid)
        avg_unsupp  = sum(r["metrics"]["unsupported_ratio"] for r in valid) / len(valid)
        avg_overlap = sum(r["metrics"]["avg_lexical_overlap"] for r in valid) / len(valid)
        parse_errs  = sum(1 for r in subset if r["metrics"]["parse_error"])
        print(f"{dataset} (n={len(valid)}, parse_errors={parse_errs})")
        print(f"  avg citation_coverage   = {avg_cov:.3f}")
        print(f"  avg unsupported_ratio   = {avg_unsupp:.3f}")
        print(f"  avg lexical_overlap     = {avg_overlap:.3f}")
        print()

    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
