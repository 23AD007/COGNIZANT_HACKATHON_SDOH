from pathlib import Path
import json
import sys
import joblib
import pandas as pd

from src.modeling.train_member_risk_model import ModelArtifact


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
    / "member_risk_model.pkl"
)

SCHEMA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
    / "member_risk_feature_schema.json"
)


# ------------------------------------------------------------------
# Compatibility for the existing pickle
# ------------------------------------------------------------------
# The model was originally saved when ModelArtifact was in __main__.
# Make that class available under __main__ before loading the pickle.
sys.modules["__main__"].ModelArtifact = ModelArtifact


# ------------------------------------------------------------------
# Load existing trained model
# ------------------------------------------------------------------
model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------------
# Load exact feature schema used during training
# ------------------------------------------------------------------
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)


FEATURE_COLUMNS = schema["feature_columns"]


def predict_member_risk(data: dict) -> float:
    """
    Predict risk using the already-trained model.

    This function does NOT train or modify the model.
    """

    X = pd.DataFrame([data])

    # Use exactly the same feature order as training.
    X = X[FEATURE_COLUMNS]

    probability = float(
        model.pipeline.predict_proba(X)[0, 1]
    )

    return probability
def get_risk_band(probability: float) -> str:
    if probability <= 0.20:
        return "Very Low"
    elif probability <= 0.40:
        return "Low"
    elif probability <= 0.60:
        return "Medium"
    elif probability <= 0.80:
        return "High"
    else:
        return "Very High"