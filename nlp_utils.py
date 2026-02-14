import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Simple text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SimilarityEngine:
    def __init__(self, documents):
        # Clean all documents
        self.documents = [clean_text(doc) for doc in documents]

        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query, top_k=1):
        # Clean user query
        query = clean_text(query)

        # Convert query to vector
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity
        similarity_scores = cosine_similarity(query_vector, self.matrix)[0]

        # Get best match
        sorted_indices = similarity_scores.argsort()[::-1][:top_k]

        return sorted_indices, similarity_scores[sorted_indices]
