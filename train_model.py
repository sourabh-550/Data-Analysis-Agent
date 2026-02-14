import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nlp_utils import clean_text


# Paths
DATA_PATH = "data/all_tickets_processed_improved_v3.csv"
MODEL_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/tfidf.pkl"


def load_data(path):
    df = pd.read_csv(path)

    # Keep only needed columns
    df = df[["Document", "Topic_group"]]
    df = df.dropna()

    df.rename(columns={
        "Document": "text",
        "Topic_group": "category"
    }, inplace=True)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    return df


def train():
    df = load_data(DATA_PATH)

    X = df["text"]
    y = df["category"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert text to numbers
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    predictions = model.predict(X_test_vec)
    print("\nModel Performance:\n")
    print(classification_report(y_test, predictions))

    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel and vectorizer saved successfully!")


if __name__ == "__main__":
    train()
