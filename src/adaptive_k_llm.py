class AdaptiveKLLM:
    def __init__(self, llm):
        # llm is the MedRAG instance itself
        self.llm = llm

    def predict_k(self, query):
        # print("[AdaptiveKLLM] predict_k called", flush=True) 
        messages = [
            {
                "role": "system",
                "content": "You are a medical question complexity classifier. Output exactly one word."
            },
            {
                "role": "user",
                "content": (
                    "Classify the complexity of this biomedical question.\n\n"
                    f"Question: {query}\n\n"
                    "Output only one word: simple / medium / complex"
                )
            }
        ]
        response = self.llm.generate(messages)
        response = response.lower().strip()

        if "simple" in response:
            k = 8
        elif "medium" in response:
            k = 16
        else:
            k = 32

        return k