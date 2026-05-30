import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 50)

# STEP 1: Load Data
print("\n[1/6] Loading data...")
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"✓ Loaded {len(df)} employee records")

# STEP 2: Prepare Data
print("\n[2/6] Preparing data...")

# Convert target to binary
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
print(f"✓ Attrition cases: {df['Attrition'].sum()} ({df['Attrition'].mean()*100:.1f}%)")

# Feature Engineering - NEW FEATURES ADDRESSING RESEARCH GAPS
print("\n[3/6] Engineering features...")
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['SatisfactionWorkloadRatio'] = df['JobSatisfaction'] / (df['DistanceFromHome'] + 1)
df['CareerStagnation'] = ((df['YearsSinceLastPromotion'] > 3) & 
                           (df['PerformanceRating'] >= 3)).astype(int)
df['IncomeToAgeRatio'] = df['MonthlyIncome'] / df['Age']
df['ExperienceToPromotionRatio'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
print("✓ Created 5 new features")

# Encode categorical variables
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Remove unnecessary columns
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Prepare X and y
X = df.drop('Attrition', axis=1)
y = df['Attrition']

feature_names = X.columns.tolist()
print(f"✓ Total features: {len(feature_names)}")

# STEP 3: Split Data
print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# STEP 4: Handle Class Imbalance - ADDRESSING RESEARCH GAP
print("\n[5/6] Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✓ Balanced training samples: {len(X_train_balanced)}")
print(f"✓ Attrition cases after SMOTE: {y_train_balanced.sum()} (50%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# STEP 5: Train Model
print("\n[6/6] Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(X_train_scaled, y_train_balanced, verbose=False)
print("✓ Model training complete!")

# STEP 6: Evaluate Model
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 ROC-AUC Score: {auc_score:.3f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# STEP 7: Create SHAP Explainer - ADDRESSING XAI RESEARCH GAP
print("\n" + "=" * 50)
print("CREATING EXPLAINABILITY MODEL (SHAP)")
print("=" * 50)
explainer = shap.TreeExplainer(model)
print("✓ SHAP explainer created")

# STEP 8: Save Everything
print("\n" + "=" * 50)
print("SAVING MODEL FILES")
print("=" * 50)

joblib.dump(model, 'model.pkl')
print("✓ Saved: model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Saved: feature_names.pkl")

joblib.dump(explainer, 'explainer.pkl')
print("✓ Saved: explainer.pkl")

joblib.dump(le_dict, 'label_encoders.pkl')
print("✓ Saved: label_encoders.pkl")

# Save test metrics
metrics = {
    'auc_score': float(auc_score),
    'accuracy': float((y_test == y_pred).mean()),
    'confusion_matrix': cm.tolist()
}
joblib.dump(metrics, 'metrics.pkl')
print("✓ Saved: metrics.pkl")

print("\n" + "=" * 50)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 50)
print("\nNext step: Run 'python app.py' to start the API server")