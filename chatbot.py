import pandas as pd
import joblib
from nlp_utils import SimilarityEngine


class ITHelpdeskChatbot:
    def __init__(self, tickets_path):
        # Load trained ML model
        self.vectorizer = joblib.load("models/tfidf.pkl")
        self.classifier = joblib.load("models/classifier.pkl")

        # Load knowledge base
        self.df = pd.read_csv(tickets_path)

        # Create similarity engine for each category
        self.engines = {}
        for category in self.df["category"].unique():
            data = self.df[self.df["category"] == category]
            self.engines[category] = {
                "engine": SimilarityEngine(data["issue"].tolist()),
                "answers": data["resolution"].tolist()
            }

    def rule_based_category(self, text):
        text = text.lower()

        if any(word in text for word in ["wifi", "internet", "router", "network"]):
            return "Hardware"

        if any(word in text for word in ["vpn", "login", "password", "access", "folder", "drive"]):
            return "Access"

        if any(word in text for word in ["keyboard", "mouse", "printer", "screen", "usb"]):
            return "Hardware"

        if any(word in text for word in ["leave", "hr", "employee"]):
            return "HR Support"

        if any(word in text for word in ["purchase", "buy", "asset"]):
            return "Purchase"

        return None

    def predict_category(self, text):
        vec = self.vectorizer.transform([text])
        return self.classifier.predict(vec)[0]

    def fallback_message(self, category):
        email = "\n\n📧 Contact IT support: sampleitmail@gmail.com"


        return "Please contact IT support for further assistance." + email

    def get_response(self, user_query, threshold=0.7):
        # Step 1: Try rule-based detection
        category = self.rule_based_category(user_query)

        # Step 2: If no rule match, use ML
        if category is None:
            category = self.predict_category(user_query)

        # Step 3: If category not found
        if category not in self.engines:
            return {
                "message": self.fallback_message(category),
                "category": category
            }

        engine = self.engines[category]["engine"]
        answers = self.engines[category]["answers"]

        indices, scores = engine.search(user_query, top_k=1)
        similarity = scores[0]

        # Step 4: If similarity low → fallback
        if similarity < threshold:
            return {
                "message": self.fallback_message(category),
                "category": category
            }

        # Step 5: Return matched solution
        return {
            "message": answers[indices[0]],
            "category": category
        }
