import json
import joblib
from pathlib import Path

CURRENT_META_PATH = Path(r"registry/current_model.json")

_model = None
_model_info = {
    "model_name": None,
    "model_version": "local-production"
}

def load_model():
    global _model, _model_info

    if not CURRENT_META_PATH.exists():
        raise FileNotFoundError("registry/current_model.json not found")

    with open(CURRENT_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = Path(meta["model_path"])
    _model = joblib.load(model_path)
    _model_info = {
        "model_name": meta.get("model_name", "churn_prediction_model"),
        "model_version": "local-production"
    }
    return _model

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def reload_model():
    return load_model()

def get_model_info():
    return _model_info
