import argparse
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("customer_churn_training")

    df = pd.read_parquet(args.input_path)

    y = df["Churn"]
    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols),
    ])

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(args.output_dir) / "model.joblib"
        joblib.dump(pipeline, model_path)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("test_size", 0.2)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # KHONG log model artifact vao MLflow de tranh loi artifact store tren Windows
        # mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print({
            "run_id": run.info.run_id,
            "model_path": str(model_path),
            "metrics": metrics
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mlflow_uri", default="http://localhost:5000")
    args = parser.parse_args()
    main(args)