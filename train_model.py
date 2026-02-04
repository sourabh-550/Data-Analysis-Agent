"""Train and evaluate the IT Helpdesk ticket classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from nlp_utils import clean_text

DATA_PATH = Path("data/all_tickets_processed_improved_v3.csv")
MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
MODEL_PATH = MODEL_DIR / "classifier.pkl"


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load and clean the Kaggle ticket dataset.

    Args:
        path: Path to the CSV dataset.

    Returns:
        Cleaned DataFrame with columns: text, category.
    """
    df = pd.read_csv(path, usecols=["Document", "Topic_group"])
    df = df.rename(columns={"Document": "text", "Topic_group": "category"})
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3]
    df["text"] = df["text"].apply(clean_text)
    return df


def train_model(df: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Train the TF-IDF + Logistic Regression classifier."""
    features = df["text"]
    labels = df["category"]

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)

    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        multi_class="auto",
        class_weight="balanced",
    )
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_val_vec)
    print("\nValidation Classification Report")
    print("=" * 40)
    print(classification_report(y_val, y_pred))

    return vectorizer, model


def save_artifacts(vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    """Save vectorizer and model artifacts to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")


def main() -> None:
    """Run the training pipeline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please place the CSV in the data folder."
        )

    df = load_and_clean_data(DATA_PATH)
    vectorizer, model = train_model(df)
    save_artifacts(vectorizer, model)


if __name__ == "__main__":
    main()
