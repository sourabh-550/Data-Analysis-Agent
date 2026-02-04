import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SimilarityEngine:
    def __init__(self, documents):
        self.raw_docs = documents
        self.cleaned_docs = [clean_text(d) for d in documents]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_docs)

    def search(self, query, top_k=1):
        query_clean = clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])

        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        best_indices = sims.argsort()[::-1][:top_k]
        best_scores = sims[best_indices]

        return best_indices, best_scores
