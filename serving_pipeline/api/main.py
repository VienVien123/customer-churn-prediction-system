from fastapi import FastAPI
import pandas as pd

from serving_pipeline.api.schemas import PredictRequest, ReloadRequest
from serving_pipeline.api.model_loader import get_model, reload_model, get_model_info

app = FastAPI(title="Customer Churn Prediction API")

def build_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Tenure_Group"] = pd.cut(
        df["Tenure"],
        bins=[-1, 6, 12, 24, 48, 120],
        labels=["0_6", "7_12", "13_24", "25_48", "49_plus"]
    )

    df["Avg_Monthly_Spend"] = df["Total_Spend"] / (df["Tenure"] + 1)
    df["Support_to_Usage_Ratio"] = df["Support_Calls"] / (df["Usage_Frequency"] + 1)
    df["High_Payment_Delay"] = (df["Payment_Delay"] > 15).astype(int)
    df["Inactive_Recently"] = (df["Last_Interaction"] > 15).astype(int)

    df = df.rename(columns={
        "Usage_Frequency": "Usage Frequency",
        "Support_Calls": "Support Calls",
        "Payment_Delay": "Payment Delay",
        "Subscription_Type": "Subscription Type",
        "Contract_Length": "Contract Length",
        "Total_Spend": "Total Spend",
        "Last_Interaction": "Last Interaction"
    })

    return df

@app.get("/health")
def health():
    info = get_model_info()
    return {
        "status": "ok",
        "model_loaded": True,
        **info
    }

@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    df = pd.DataFrame([req.model_dump()])
    df = build_features_for_inference(df)

    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "prediction_label": "Yes" if pred == 1 else "No",
        "churn_probability": prob
    }

@app.post("/reload-model")
def reload_model_endpoint(req: ReloadRequest):
    reload_model()
    return {"status": "reloaded"}
