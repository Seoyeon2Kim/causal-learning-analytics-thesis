# src/featureengineering.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

_EXCLUDE = {"student_id", "week", "dropout", "failure",
            "final_result", "code_module", "code_presentation"}

def numeric_feature_columns(df: pd.DataFrame) -> list:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in _EXCLUDE]

def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

# ── aliases ──────────────────────────────────────────────
numericfeaturecolumns = numeric_feature_columns
makepreprocessor      = make_preprocessor