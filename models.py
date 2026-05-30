from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    department = Column(String(100))
    job_role = Column(String(100))
    monthly_income = Column(Integer)
    years_at_company = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer)
    risk_probability = Column(Float)
    risk_level = Column(String(20))

    created_at = Column(DateTime, default=datetime.utcnow)


class ShapExplanation(Base):
    __tablename__ = "shap_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer)
    feature_name = Column(String(100))
    impact = Column(Float)
    direction = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)


class RecommendedAction(Base):
    __tablename__ = "recommended_actions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer)
    action = Column(String(255))
    priority = Column(String(50))
    description = Column(String(255))

    created_at = Column(DateTime, default=datetime.utcnow)
