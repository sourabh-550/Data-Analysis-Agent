import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")

# 2. Keep only required columns
df = df[["Document", "Topic_group"]]

# 3. Rename for convenience
df.columns = ["text", "category"]

# 4. Drop missing values
df = df.dropna()

# 5. Split into X and y
X = df["text"].astype(str)
y = df["category"].astype(str)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train classifier
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 10. Save model and vectorizer
joblib.dump(vectorizer, "models/tfidf.pkl")
joblib.dump(model, "models/classifier.pkl")

print("✅ Model and vectorizer saved in models/ folder")
