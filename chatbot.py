import pandas as pd
import joblib
from nlp_utils import SimilarityEngine

class ITHelpdeskChatbot:
    def __init__(self, tickets_path):
        # Load ML model
        self.vectorizer = joblib.load("models/tfidf.pkl")
        self.classifier = joblib.load("models/classifier.pkl")

        # Load knowledge base with solutions
        self.df = pd.read_csv(tickets_path)

        # Build similarity engine per category
        self.engines = {}
        for category in self.df["category"].unique():
            subset = self.df[self.df["category"] == category]
            questions = subset["issue"].tolist()
            self.engines[category] = {
                "engine": SimilarityEngine(questions),
                "answers": subset["resolution"].tolist()
            }

        # 🔹 Simple memory
        self.last_category = None
        self.last_issue = None

    def predict_category(self, text):
        vec = self.vectorizer.transform([text])
        return self.classifier.predict(vec)[0]

    def is_followup(self, text):
        # Very simple check for follow-up messages
        followup_keywords = [
            "still", "not working", "didn't work", "did not work",
            "doesn't work", "does not work", "i tried", "already tried",
            "same issue", "again"
        ]
        text_lower = text.lower()
        return any(k in text_lower for k in followup_keywords)

    def get_response(self, user_query, threshold=0.2):
        # Check if this looks like a follow-up
        if self.last_issue is not None and self.is_followup(user_query):
            # Combine previous issue with current message
            combined_query = self.last_issue + " " + user_query
            category = self.last_category
        else:
            combined_query = user_query
            category = self.predict_category(user_query)

        # Save memory
        self.last_category = category
        self.last_issue = combined_query

        if category not in self.engines:
            return f"I think this is a **{category}** issue, but I don't have solutions for this category yet."

        engine = self.engines[category]["engine"]
        answers = self.engines[category]["answers"]

        indices, scores = engine.search(combined_query, top_k=1)

        if scores[0] < threshold:
            return f"I think this is a **{category}** issue, but I couldn't find a good solution. Please contact IT support."

        return f"**Predicted Category:** {category}\n\n**Solution:** {answers[indices[0]]}"
