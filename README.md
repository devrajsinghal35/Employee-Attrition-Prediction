# 🚀 Employee Attrition Management System

An end-to-end **AI-powered HR analytics platform** that predicts employee attrition, explains the reasons behind it, and provides an interactive dashboard with an LLM-powered assistant.

---

## 📌 Overview

Employee attrition is a critical challenge for organizations. This project leverages **Machine Learning + Explainable AI + LLMs** to:

* 🔍 Predict whether an employee is likely to leave
* 📊 Visualize attrition risk
* 🧠 Explain predictions using SHAP
* 🤖 Provide HR insights via chatbot

---

## 🧠 Key Features

### 🔹 1. Attrition Prediction

* Predicts probability of employee leaving
* Uses trained ML model (XGBoost)

### 🔹 2. Explainable AI (SHAP)

* Shows **feature contribution** for each prediction
* Helps understand *why* an employee may leave

### 🔹 3. Interactive Dashboard (Streamlit)

* User-friendly form with 30+ features
* Visual outputs:

  * Attrition probability
  * Bar charts
  * Feature contribution pie chart

### 🔹 4. LLM-powered HR Chatbot

* Built using Groq API
* Answers HR-related queries
* Uses prediction context for smarter insights

### 🔹 5. Model Performance Metrics

* Accuracy
* ROC-AUC
* Displayed inside UI

---

## 🏗️ Tech Stack

### 🔹 Backend

* FastAPI
* Uvicorn

### 🔹 Frontend

* Streamlit

### 🔹 Machine Learning

* Scikit-learn
* XGBoost
* SMOTE (class imbalance handling)

### 🔹 Explainability

* SHAP

### 🔹 LLM Integration

* Groq (LLaMA 3.1)

### 🔹 Deployment

* Render (Backend API)
* Streamlit Cloud (Frontend)

---

## 📁 Project Structure

```
Employee-Attrition-Prediction/

├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit UI
├── prediction_service.py   # ML inference logic
├── feature_pipeline.py     # Feature engineering + preprocessing
├── llm_helper.py           # Chatbot logic (Groq)
├── schemas.py              # API request schemas
├── train_model.py          # Model training script

├── model.pkl               # Trained ML model
├── scaler.pkl              # Feature scaler
├── feature_names.pkl       # Feature order
├── label_encoders.pkl      # Encoders
├── explainer.pkl           # SHAP explainer
├── metrics.pkl             # Model metrics

├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Set Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4️⃣ Run Backend

```bash
uvicorn app:app --reload
```

---

### 5️⃣ Run Frontend

```bash
streamlit run streamlit_app.py
```

---

## 🚀 Deployment

### 🔹 Backend (Render)

* Deploy FastAPI app
* Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

### 🔹 Frontend (Streamlit Cloud)

* Connect GitHub repo
* Select `streamlit_app.py`
* Add secrets:

```
GROQ_API_KEY=your_key
```

---

## 📊 Model Details

* Model: **XGBoost Classifier**
* Handles class imbalance using **SMOTE**
* Evaluated using:

  * Accuracy
  * ROC-AUC
  * Classification Report

---

## 🔬 Feature Engineering

Custom features created to improve prediction:

* `IncomeToAgeRatio`
* `ExperienceToPromotionRatio`
* `SatisfactionWorkloadRatio`
* `PromotionGap`
* `CareerStagnation`

These capture:

* Salary fairness
* Career growth
* Work-life balance
* Promotion delays

---

## 🧠 Explainability

Uses SHAP to:

* Identify top contributing features
* Visualize impact using pie chart
* Provide interpretable AI outputs

---

## 📸 Screenshots

*(Add screenshots here after deployment)*

* Dashboard UI
* Prediction output
* SHAP feature contribution
* Chatbot interaction

---

## 📈 Example Output

```
Attrition Risk: 65%

Top Factors:
- Low Monthly Income
- High Overtime
- Career Stagnation
```

---

## 🔥 Future Improvements

* 🎯 Improve recall for attrition class
* 📊 Add ROC curve visualization
* 🧠 Better SHAP explanations (top reasons text)
* 🎨 UI/UX enhancements
* 📡 Real-time employee monitoring

---

## 👨‍💻 Author

**Devraj Singhal**

* 📍 India
* 💻 Cyber Security Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

---
