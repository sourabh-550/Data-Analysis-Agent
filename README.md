## 🚀 Live Demo

🔗 Deployed App: https://it-hepldesk-chatbot-sourabh-saxena.streamlit.app/

# 💻 IT Helpdesk Chatbot (ML + NLP + Streamlit)

An **IT Helpdesk Chatbot** that provides quick troubleshooting guidance for common IT issues such as **WiFi/network problems, keyboard/printer issues, login & access issues, HR requests, and asset/purchase queries**.  
The system uses a **hybrid approach**: **Machine Learning** for category prediction + **NLP similarity search** over a **clean knowledge base** for solution retrieval, all wrapped in a **Streamlit chat UI**.

> Built as part of my learning journey in the **GENz AI Programme**.

Live Woking deployed chatbot
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
> A clean, curated .json is used for solution retrieval (to avoid noisy matches).

---

## ⚙️ Installation

## 🚀 Getting Started

Clone the repository:

git clone https://github.com/sourabh-550/IT-Helpdesk-chatbot.git
cd it-helpdesk-chatbot

---

## ⚙️ Create and Activate Virtual Environment (Optional but Recommended)

python -m venv venv
venv\Scripts\activate

---

## 📦 Install Dependencies

pip install -r requirements.txt

---

## 🧪 Train the Model (One-Time)

Dataset Link: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

Make sure your Kaggle dataset is in `data/` as:

all_tickets_processed_improved_v3.csv

Run:

python train_model.py

This will create:

models/tfidf.pkl  
models/classifier.pkl  

---

## ▶️ Run the App

streamlit run app.py

---

## 💬 Example Queries

- Wifi is not working  
- Keyboard is not working  
- Cannot login to email  
- VPN is not connecting  
- I need a new laptop  
- My leave is not updated  

---

## 📊 Why This Project

- Uses a real-world Kaggle dataset for training  
- Demonstrates a complete ML pipeline:  
  Data → Training → Saving Model → Loading → Prediction  
- Shows how hybrid systems (rules + ML + NLP search) are used in real enterprise applications  
- Focuses on practical usability, not just model accuracy  
- Moves beyond notebooks to a real, interactive application  

---

## 🎯 Future Improvements

- Multi-step guided troubleshooting (Step 1 → Did it work? → Step 2)  
- User feedback system (Was this helpful? Yes/No)  
- Expand and refine the knowledge base   
- Role-based routing (Network team, HR team, Admin team, etc.)  
- Analytics dashboard for common issues  

---

## 🙌 Acknowledgements

- Kaggle – IT Service Ticket Classification Dataset  
- GENz AI Programme – for structured learning and guidance  
- scikit-learn & Streamlit communities  

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub and feel free to fork or contribute


