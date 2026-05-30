from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import numpy as np
import joblib

from database import get_db
from models import Employee, Prediction, ShapExplanation, RecommendedAction
from schemas import EmployeeInput

app = FastAPI(
    title="Employee Attrition Prediction System",
    version="2.0",
    description="Enterprise-grade Employee Attrition Prediction with Explainable AI"
)

# --------------------------------------------------
# CORS CONFIGURATION (OPTION B – STRICT & SAFE)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "null"  # required for file:// origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load training artifacts (SOURCE OF TRUTH)
# --------------------------------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = joblib.load("explainer.pkl")
feature_names = joblib.load("feature_names.pkl")
label_encoders = joblib.load("label_encoders.pkl")

CATEGORICAL_COLS = set(label_encoders.keys())

# --------------------------------------------------
# Helper logic
# --------------------------------------------------

def risk_bucket(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def encode_categorical(feature: str, value: str):
    encoder = label_encoders[feature]
    if value not in encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Unseen categorical value '{value}' for feature '{feature}'"
        )
    return encoder.transform([value])[0]


def build_feature_vector(data: dict):
    # --- engineered features (MATCH train_model.py) ---
    data["PromotionGap"] = data["YearsAtCompany"] - data["YearsSinceLastPromotion"]
    data["SatisfactionWorkloadRatio"] = (
        data["JobSatisfaction"] / (data["DistanceFromHome"] + 1)
    )
    data["CareerStagnation"] = int(
        data["YearsSinceLastPromotion"] > 3 and data["PerformanceRating"] >= 3
    )
    data["IncomeToAgeRatio"] = data["MonthlyIncome"] / data["Age"]
    data["ExperienceToPromotionRatio"] = (
        data["YearsAtCompany"] / (data["YearsSinceLastPromotion"] + 1)
    )

    feature_row = {}

    for feature in feature_names:
        if feature not in data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature '{feature}'"
            )

        value = data[feature]

        if feature in CATEGORICAL_COLS:
            feature_row[feature] = encode_categorical(feature, value)
        else:
            feature_row[feature] = float(value)

    X = np.array([[feature_row[f] for f in feature_names]])
    return X, feature_row


def extract_top_factors(shap_values, feature_values, top_k=5):
    combined = list(zip(feature_names, shap_values, feature_values.values()))
    combined.sort(key=lambda x: abs(x[1]), reverse=True)

    top = []
    for feature, impact, value in combined[:top_k]:
        top.append({
            "factor": feature,
            "impact": float(impact),
            "direction": "INCREASES RISK" if impact > 0 else "DECREASES RISK",
            "value": float(value)
        })

    return top


def action_engine(top_factors):
    actions = []

    for f in top_factors:
        if f["factor"] == "MonthlyIncome" and f["impact"] > 0:
            actions.append({
                "action": "Compensation Review",
                "priority": "High",
                "description": "Low income is a major attrition driver."
            })

        if f["factor"] == "YearsSinceLastPromotion" and f["impact"] > 0:
            actions.append({
                "action": "Promotion & Career Path Discussion",
                "priority": "Medium",
                "description": "Career stagnation detected."
            })

        if f["factor"] == "OverTime" and f["impact"] > 0:
            actions.append({
                "action": "Workload Optimization",
                "priority": "High",
                "description": "Excessive overtime contributing to burnout."
            })

    return actions

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.post("/predict")
def predict_attrition(
    payload: EmployeeInput,
    db: Session = Depends(get_db)
):
    data = payload.dict()

    # 1️⃣ Build feature vector
    X, feature_value_map = build_feature_vector(data)
    X_scaled = scaler.transform(X)

    # 2️⃣ Predict
    probability = float(model.predict_proba(X_scaled)[0][1])
    risk_level = risk_bucket(probability)

    # 3️⃣ Explainability
    shap_values = explainer.shap_values(X_scaled)[0]
    top_factors = extract_top_factors(shap_values, feature_value_map)
    actions = action_engine(top_factors)

    # 4️⃣ Persist Employee snapshot
    employee = Employee(
        age=data["Age"],
        department=data["Department"],
        job_role=data["JobRole"],
        monthly_income=data["MonthlyIncome"],
        years_at_company=data["YearsAtCompany"]
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)

    # 5️⃣ Persist Prediction
    prediction = Prediction(
        employee_id=employee.id,
        risk_probability=probability,
        risk_level=risk_level
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # 6️⃣ Persist SHAP
    for feature, impact in zip(feature_names, shap_values):
        db.add(
            ShapExplanation(
                prediction_id=prediction.id,
                feature_name=feature,
                impact=float(impact),
                direction="Positive" if impact > 0 else "Negative"
            )
        )

    # 7️⃣ Persist Actions
    for action in actions:
        db.add(
            RecommendedAction(
                prediction_id=prediction.id,
                action=action["action"],
                priority=action["priority"],
                description=action["description"]
            )
        )

    db.commit()

    # 8️⃣ Frontend-aligned response
    return {
        "employee_id": employee.id,
        "risk_probability": probability,
        "risk_level": risk_level,
        "top_factors": top_factors,
        "recommended_actions": actions
    }


@app.get("/predictions/{employee_id}")
def prediction_history(employee_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Prediction)
        .filter(Prediction.employee_id == employee_id)
        .all()
    )
