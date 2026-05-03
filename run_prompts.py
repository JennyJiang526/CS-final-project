import json
import os
import time
from pathlib import Path

from openai import OpenAI


def _load_dotenv() -> None:
    """从同目录 .env 读取 OPENAI_API_KEY（勿提交 .env）。"""
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

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "medical_qa_1000.jsonl"
OUTPUT_DIR = "results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("请设置环境变量 OPENAI_API_KEY 或在同目录创建 .env 写入 OPENAI_API_KEY=...")

client = OpenAI(api_key=API_KEY)

# ============================================================
# PROMPT TEMPLATES
# ============================================================
def get_baseline_prompt(record):
    question = record["question"]
    if record["dataset"] == "PubMedQA":
        return f"""Answer the following biomedical question with yes, no, or maybe.
Your first word MUST be yes, no, or maybe.
Question: {question}
Answer:"""
    elif record["dataset"] == "MedQA-US":
        return f"""Answer the following medical question. Start your answer with the letter of the correct option (A, B, C, D, or E).
Question: {question}
Answer:"""
    else:
        return f"""Answer the following medical question concisely.
Question: {question}
Answer:"""


def get_cot_prompt(record):
    question = record["question"]
    if record["dataset"] == "PubMedQA":
        return f"""Answer the following biomedical question step by step, then give a final decision.
Question: {question}
Work through this step by step:
1. Key concepts: What is the study about?
2. Reasoning: What do the results suggest?
3. Final decision: yes, no, or maybe (this must be your last line, format: "Final decision: yes/no/maybe")"""
    elif record["dataset"] == "MedQA-US":
        return f"""Answer the medical question step by step, then state the correct letter.
Question: {question}
Work through this step by step:
1. Key concepts: What medical concepts are relevant?
2. Reasoning: Apply those concepts to the question.
3. Conclusion: The answer is [letter]."""
    else:
        return f"""Answer the medical question step by step.
Question: {question}
Work through this step by step:
1. Key concepts: What medical concepts are relevant?
2. Reasoning: Apply those concepts.
3. Conclusion: State your final answer clearly."""


def get_meta_prompt(record):
    question = record["question"]
    if record["dataset"] == "PubMedQA":
        return f"""You are a biomedical research expert.
Answer the following question with yes, no, or maybe. Your first word MUST be yes, no, or maybe.
Question: {question}
Answer:"""
    elif record["dataset"] == "MedQA-US":
        return f"""You are a medical expert. Answer the following question. Start with the correct letter (A-E).
Question: {question}
Answer:"""
    else:
        return f"""You are experts having biomedical and clinical knowledge.
Answer the following medical question:
Question: {question}
Answer:"""

PROMPT_FUNCS = {
    "baseline": get_baseline_prompt,
    "cot": get_cot_prompt,
    "meta": get_meta_prompt,
}

# ============================================================
# GPT-4o CALL
# ============================================================
def call_gpt4o(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=42,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            time.sleep(5)
    return None

# ============================================================
# MAIN RUNNER
# ============================================================
def run(prompt_name):
    output_file = os.path.join(OUTPUT_DIR, f"{prompt_name}_results.jsonl")
    
    # 已跑过的题目跳过（断点续跑）
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except:
                    pass
    print(f"\n=== Running: {prompt_name} | Already done: {len(done_ids)} ===")

    get_prompt = PROMPT_FUNCS[prompt_name]

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            record = json.loads(line.strip())
            record_id = idx  # 用行号作ID

            if record_id in done_ids:
                continue

            question = record["question"]
            prompt = get_prompt(record)

            print(f"  [{idx+1}/1000] dataset={record['dataset']}", end=" ... ", flush=True)
            answer = call_gpt4o(prompt)

            if answer is None:
                print("FAILED")
                continue

            result = {
                "id": record_id,
                "dataset": record["dataset"],
                "question": question,
                "ground_truth": record["answer"],
                "prompt_type": prompt_name,
                "model_answer": answer,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            print("OK")

            time.sleep(0.5)  # rate limit保护

    print(f"=== Done: {prompt_name} → {output_file} ===\n")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    for prompt_name in ["baseline", "cot", "meta"]:
        run(prompt_name)
