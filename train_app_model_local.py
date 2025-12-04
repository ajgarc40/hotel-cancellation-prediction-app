import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load data
df = pd.read_csv("hotel_bookings.csv")

# 2. Features and target, matching the Databricks notebook
feature_cols = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "adults",
    "children",
    "babies",
    "deposit_type",
    "market_segment",
    "customer_type",
]

target_col = "is_canceled"

X = df[feature_cols].copy()
y = df[target_col].copy()

# 3. Numeric and categorical features
numeric_features = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "adults",
    "children",
    "babies",
]

categorical_features = [
    "deposit_type",
    "market_segment",
    "customer_type",
]

# 4. Preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Random Forest model for the app
rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
)

rf_app_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ]
)

# 6. Train and quick evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

rf_app_pipeline.fit(X_train, y_train)

y_pred = rf_app_pipeline.predict(X_test)
print("Classification report on local test split:")
print(classification_report(y_test, y_pred))

# 7. Save as model.pkl for Streamlit
joblib.dump(rf_app_pipeline, "model.pkl")
print("Saved model to model.pkl")
