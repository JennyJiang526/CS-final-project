# 三条基线所用 Prompt（与 `run_prompts.py` 一致）

说明：实验在 `run_prompts.py` 中分别以 `baseline`、`cot`、`meta` 三种 `prompt_type` 调用 GPT-4o；每条记录按 `dataset` 字段分支选择下列模板之一。占位符 `{question}` 在代码中为 `record["question"]` 的实际文本。

---

## 1. Baseline（`get_baseline_prompt`）

### PubMedQA

```
Answer the following biomedical question with yes, no, or maybe.
Your first word MUST be yes, no, or maybe.
Question: {question}
Answer:
```

### MedQA-US

```
Answer the following medical question. Start your answer with the letter of the correct option (A, B, C, D, or E).
Question: {question}
Answer:
```

### 其他数据集（BioASQ、MediQA 等）

```
Answer the following medical question concisely.
Question: {question}
Answer:
```

---

## 2. CoT（`get_cot_prompt`）

### PubMedQA

```
Answer the following biomedical question step by step, then give a final decision.
Question: {question}
Work through this step by step:
1. Key concepts: What is the study about?
2. Reasoning: What do the results suggest?
3. Final decision: yes, no, or maybe (this must be your last line, format: "Final decision: yes/no/maybe")
```

### MedQA-US

```
Answer the medical question step by step, then state the correct letter.
Question: {question}
Work through this step by step:
1. Key concepts: What medical concepts are relevant?
2. Reasoning: Apply those concepts to the question.
3. Conclusion: The answer is [letter].
```

### 其他数据集

```
Answer the medical question step by step.
Question: {question}
Work through this step by step:
1. Key concepts: What medical concepts are relevant?
2. Reasoning: Apply those concepts.
3. Conclusion: State your final answer clearly.
```

---

## 3. Meta（`get_meta_prompt`）

### PubMedQA

```
You are a biomedical research expert.
Answer the following question with yes, no, or maybe. Your first word MUST be yes, no, or maybe.
Question: {question}
Answer:
```

### MedQA-US

```
You are a medical expert. Answer the following question. Start with the correct letter (A-E).
Question: {question}
Answer:
```

### 其他数据集

```
You are experts having biomedical and clinical knowledge.
Answer the following medical question:
Question: {question}
Answer:
```

---

## 调用参数（节选，见 `run_prompts.py`）

- 模型：`gpt-4o`
- 消息：`[{"role": "user", "content": prompt}]`
- `temperature=0`，`seed=42`，`max_tokens=2048`
