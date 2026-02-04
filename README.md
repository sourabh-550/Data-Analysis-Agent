# 💻 IT Helpdesk Chatbot (ML + NLP + Streamlit)

An **IT Helpdesk Chatbot** that provides quick troubleshooting guidance for common IT issues such as **WiFi/network problems, keyboard/printer issues, login & access issues, HR requests, and asset/purchase queries**.  
The system uses a **hybrid approach**: **Machine Learning** for category prediction + **NLP similarity search** over a **clean knowledge base** for solution retrieval, all wrapped in a **Streamlit chat UI**.

> Built as part of my learning journey in the **GENz AI Programme**.

---

## 🚀 Features

- 💬 Chat-based UI built with Streamlit  
- 🧠 ML-powered issue classification (trained on a real Kaggle IT Service Ticket dataset)  
- 🔎 NLP solution retrieval using **TF-IDF + Cosine Similarity**  
- 🧩 Hybrid system: **Rule-based routing + ML fallback** for better accuracy  
- 🛟 Smart fallback troubleshooting when no close match is found  
- 📧 Shows IT contact email when escalation is needed  
- 🏷️ Displays detected category for transparency  
- 🧼 Clean, curated knowledge base for reliable answers  

---

## 🛠️ Tech Stack

- Python  
- Streamlit (Web UI)  
- pandas, NumPy  
- scikit-learn  
  - TF-IDF Vectorizer  
  - Logistic Regression (for classification)  
- joblib (model persistence)  
- NLP techniques: text cleaning, vectorization, similarity search  

---

## 🧠 How It Works (Architecture)

1. User enters an issue in the chat UI.  
2. Rule-based router checks for obvious keywords (e.g., wifi, keyboard, vpn).  
3. If no rule matches, the ML classifier predicts the issue category.  
4. The bot filters the knowledge base by category.  
5. It runs TF-IDF + Cosine Similarity to find the closest matching issue.  
6. If similarity is high → returns the best solution.  
7. If similarity is low → shows category-specific troubleshooting steps and an IT contact email.  
8. The UI shows the answer and the detected category.

> Kaggle dataset is used for training the classifier.  
> A clean, curated CSV is used for solution retrieval (to avoid noisy matches).

---

## 📁 Project Structure

it-helpdesk-chatbot/
│
├── app.py                  # Streamlit UI
├── chatbot.py              # Core logic (rules + ML + similarity search)
├── nlp_utils.py            # Text cleaning + SimilarityEngine
├── train_model.py          # Train ML classifier on Kaggle dataset
│
├── data/
│   ├── ticket_history.csv  # Clean knowledge base (issue, resolution, category)
│   └── all_tickets_processed_improved_v3.csv  # Kaggle dataset (for training)
│
├── models/
│   ├── tfidf.pkl           # Saved TF-IDF vectorizer
│   └── classifier.pkl      # Saved ML model
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation

1. Clone the repository:
