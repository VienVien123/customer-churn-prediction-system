import pandas as pd
import json
import sys
from pathlib import Path

REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn"
]

def main(input_path, output_report):
    df = pd.read_csv(input_path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_cols,
        "null_counts": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "is_valid": len(missing_cols) == 0
    }

    Path(output_report).parent.mkdir(parents=True, exist_ok=True)

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(report)

    if not report["is_valid"]:
        raise ValueError(f"Missing columns: {missing_cols}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
