class AdaptiveKLLM:
    def __init__(self, llm):
        # llm is the MedRAG instance itself
        self.llm = llm

    def predict_k(self, query):
        # print("[AdaptiveKLLM] predict_k called", flush=True) 
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical question complexity classifier.\n"
                    "Your task is to determine how much external knowledge and reasoning is required.\n\n"
                    
                    "Definitions:\n"
                    "- simple: requires a single fact or direct recall (e.g., definition, basic concept)\n"
                    "- medium: requires combining 2–3 pieces of information or basic reasoning\n"
                    "- complex: requires multi-step reasoning, mechanism understanding, or comparison between entities\n\n"
                    
                    "Output exactly one word: simple, medium, or complex.\n"
                    "Do not output anything else."
                )
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