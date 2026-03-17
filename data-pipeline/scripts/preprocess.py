import pandas as pd
import sys
from pathlib import Path

def main(input_path, output_path):
    df = pd.read_csv(input_path)

    df.columns = [c.strip() for c in df.columns]

    df = df.dropna(how="all")
    df = df.drop_duplicates()

    if "Churn" in df.columns:
        if df["Churn"].dtype == "object":
            df["Churn"] = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
            )

    numeric_cols = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna("unknown_id").astype(str)

    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        df["Churn"] = df["Churn"].astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print({
        "processed_path": output_path,
        "rows": int(len(df)),
        "columns": list(df.columns)
    })

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
