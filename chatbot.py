import pandas as pd
import joblib
from nlp_utils import SimilarityEngine


class ITHelpdeskChatbot:
    def __init__(self, tickets_path):
        # Load ML model
        self.vectorizer = joblib.load("models/tfidf.pkl")
        self.classifier = joblib.load("models/classifier.pkl")

        # Load clean knowledge base
        self.df = pd.read_csv(tickets_path)

        # Build similarity engines per category
        self.engines = {}
        for category in self.df["category"].unique():
            subset = self.df[self.df["category"] == category]
            questions = subset["issue"].tolist()
            self.engines[category] = {
                "engine": SimilarityEngine(questions),
                "answers": subset["resolution"].tolist(),
            }

    # ---------- Keyword routing (rules first) ----------
    def rule_based_category(self, text):
        t = text.lower()

        if any(k in t for k in ["wifi", "internet", "network", "router"]):
            return "Hardware"
        if any(k in t for k in ["vpn", "login", "password", "access", "permission", "folder", "drive"]):
            return "Access"
        if any(k in t for k in ["keyboard", "mouse", "printer", "bluetooth", "screen", "usb", "laptop"]):
            return "Hardware"
        if any(k in t for k in ["leave", "hr", "onboarding", "employee"]):
            return "HR Support"
        if any(k in t for k in ["purchase", "buy", "asset", "device request", "laptop request"]):
            return "Purchase"

        return None  # no rule matched

    # ---------- ML prediction ----------
    def predict_category_ml(self, text):
        vec = self.vectorizer.transform([text])
        pred = self.classifier.predict(vec)[0]

        # Confidence from predict_proba if available
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(vec)[0]
            conf = float(probs.max())
        else:
            conf = 0.5  # fallback if model has no probabilities

        return pred, conf

    # ---------- Fallback messages ----------
    def fallback_message(self, category):
        contact_line = "\n\n📧 You can contact IT support at: **sampleitmail@gmail.com**"

        fallback = {
            "Hardware": (
                "This looks like a **Hardware/Network** issue. Try:\n"
                "1) Restart the device and router\n"
                "2) Check cables and connections\n"
                "3) Update or reinstall drivers\n"
                "4) Try again and test on another device"
                + contact_line
            ),
            "Access": (
                "This looks like an **Access** issue. Try:\n"
                "1) Check username/password\n"
                "2) Reset your password\n"
                "3) Check permissions or account status\n"
                "4) Contact IT admin if needed"
                + contact_line
            ),
            "HR Support": (
                "This looks like an **HR Support** issue. Try:\n"
                "1) Check HR portal\n"
                "2) Verify your request details\n"
                "3) Contact HR team"
                + contact_line
            ),
            "Purchase": (
                "This looks like a **Purchase/Asset** request. Try:\n"
                "1) Check request or PO status\n"
                "2) Verify approvals\n"
                "3) Contact procurement team"
                + contact_line
            ),
        }

        return fallback.get(
            category,
            "Please contact IT support for further assistance." + contact_line
        )

    # ---------- Main response ----------
    def get_response(self, user_query, threshold=0.7):
        # 1) Try rule-based routing first
        rule_cat = self.rule_based_category(user_query)

        if rule_cat is not None:
            category = rule_cat
            category_confidence = 1.0  # rules are treated as confident
        else:
            # 2) Use ML
            category, category_confidence = self.predict_category_ml(user_query)

        # 3) If we don't have this category in KB
        if category not in self.engines:
            return {
                "message": self.fallback_message(category),
                "category": category,
                "category_confidence": float(category_confidence),
                "similarity_score": 0.0,
            }

        engine = self.engines[category]["engine"]
        answers = self.engines[category]["answers"]

        indices, scores = engine.search(user_query, top_k=1)
        similarity = float(scores[0])

        # 4) If similarity is too low → fallback
        if similarity < threshold:
            return {
                "message": self.fallback_message(category),
                "category": category,
                "category_confidence": float(category_confidence),
                "similarity_score": similarity,
            }

        # 5) Good match → return solution
        return {
            "message": answers[indices[0]],
            "category": category,
            "category_confidence": float(category_confidence),
            "similarity_score": similarity,
        }
