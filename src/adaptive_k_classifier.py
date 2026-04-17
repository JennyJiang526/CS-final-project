import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------- feature extraction ----------
def extract_features(query: str):
    query_lower = query.lower()

    return [
        len(query.split()),  # length
        int(bool(re.search(r"\b(compare|vs|versus)\b", query_lower))),
        int(bool(re.search(r"\b(mechanism|pathway|process)\b", query_lower))),
        int(" and " in query_lower),
        int(" or " in query_lower),
        int(bool(re.search(r"\bwhy\b|\bhow\b", query_lower)))
    ]


# ---------- training ----------
def train_classifier(train_data, save_path="classifier.pkl"):
    """
    train_data: list of (query, label)
    label: 0 (simple), 1 (medium), 2 (complex)
    """
    X = [extract_features(q) for q, _ in train_data]
    y = [label for _, label in train_data]

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    joblib.dump(clf, save_path)
    print(f"Classifier saved to {save_path}")


# ---------- inference ----------
class AdaptiveKClassifier:
    def __init__(self, model_path="classifier.pkl"):
        self.clf = joblib.load(model_path)

    def predict_k(self, query):
        features = np.array(extract_features(query)).reshape(1, -1)
        level = self.clf.predict(features)[0]

        if level == 0:
            return 16
        elif level == 1:
            return 32
        else:
            return 48









# import re
# from typing import Optional

# # ── Keywords weight ─────────────
# COMPLEXITY_SIGNALS = {
#     # High Complexity：Multi-step reasoning, mechanism comparison
#     "high": [
#         r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b",
#         r"\bmechanism\b", r"\bpathophysiology\b",
#         r"\bmanagement of\b", r"\btreatment of\b",
#         r"\bcomplication",  r"\bdifferential",
#         r"\bwhich of the following.*best",
#         r"\bmost likely.*cause", r"\bexcept\b",
#     ],
#     # Medium Complexity：Single Diagnosis/Pharmacology
#     "medium": [
#         r"\bdiagnosis\b", r"\bdiagnose\b",
#         r"\bdrug\b", r"\bmedication\b", r"\bdose\b",
#         r"\bsymptom", r"\bsign\b", r"\bfinding",
#         r"\btest\b",  r"\bexam\b",
#     ],
#     # Low Complexity: Definitions/Facts
#     "low": [
#         r"\bwhat is\b", r"\bdefine\b", r"\bwhich\b",
#         r"\bnormal value\b", r"\bunit\b",
#     ],
# }

# K_MAP = {1: 8, 2: 16, 3: 32, 4: 48, 5: 64}


# def estimate_complexity(question: str, options: Optional[str] = None) -> int:
#     """
#     Returns a difficulty score on a scale of 1 to 5.
#     question : Question text
#     options  : String of options (if any), used to assist in evaluation
#     """
#     text = (question + " " + (options or "")).lower()
#     score = 3  # In default: Medium

#     # ── Mapping ────────────────────────────────────────────
#     high_hits = sum(
#         1 for p in COMPLEXITY_SIGNALS["high"] if re.search(p, text)
#     )
#     low_hits = sum(
#         1 for p in COMPLEXITY_SIGNALS["low"] if re.search(p, text)
#     )

#     score += min(high_hits, 2)   
#     score -= min(low_hits,  2)   

#     # ── Length Inspiraton ────────────────────────────────────────────
#     word_count = len(text.split())
#     if word_count > 80:
#         score += 1
#     elif word_count < 20:
#         score -= 1

#     # ── Number of options: 4 options are more complex than 2 options ──────────────────────────
#     if options:
#         opt_count = len(re.findall(r"\b[A-E]\b\.?\s", options))
#         if opt_count >= 4:
#             score += 1

#     return max(1, min(5, score))


# def get_adaptive_k(
#     question: str,
#     options: Optional[str] = None,
#     k_map: dict = K_MAP,
#     override_k: Optional[int] = None,
# ) -> tuple[int, int]:
#     """
#     Returns (adaptive_k, complexity_score).
#     If override_k is not None, a fixed value is used directly (for backward compatibility).
#     """
#     if override_k is not None:
#         return override_k, -1

#     complexity = estimate_complexity(question, options)
#     return k_map[complexity], complexity