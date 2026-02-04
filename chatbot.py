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

    def predict_category(self, text):
        vec = self.vectorizer.transform([text])
        return self.classifier.predict(vec)[0]

    def get_response(self, user_query, threshold=0.2):
        category = self.predict_category(user_query)

        if category not in self.engines:
            return f"I think this is a **{category}** issue, but I don't have solutions for this category yet."

        engine = self.engines[category]["engine"]
        answers = self.engines[category]["answers"]

        indices, scores = engine.search(user_query, top_k=1)

        if scores[0] < threshold:
            return f"I think this is a **{category}** issue, but I couldn't find a good solution. Please contact IT support."

        return f"**Predicted Category:** {category}\n\n**Solution:** {answers[indices[0]]}"
