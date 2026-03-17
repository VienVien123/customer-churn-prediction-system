import pandas as pd
import sys
from pathlib import Path

def main(input_path, output_path):
    df = pd.read_parquet(input_path)

    if "Tenure" in df.columns:
        df["Tenure_Group"] = pd.cut(
            df["Tenure"],
            bins=[-1, 6, 12, 24, 48, 120],
            labels=["0_6", "7_12", "13_24", "25_48", "49_plus"]
        )

    if "Total Spend" in df.columns and "Tenure" in df.columns:
        df["Avg_Monthly_Spend"] = df["Total Spend"] / (df["Tenure"] + 1)

    if "Support Calls" in df.columns and "Usage Frequency" in df.columns:
        df["Support_to_Usage_Ratio"] = df["Support Calls"] / (df["Usage Frequency"] + 1)

    if "Payment Delay" in df.columns:
        df["High_Payment_Delay"] = (df["Payment Delay"] > 15).astype(int)

    if "Last Interaction" in df.columns:
        df["Inactive_Recently"] = (df["Last Interaction"] > 15).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print({
        "feature_path": output_path,
        "rows": int(len(df)),
        "columns": list(df.columns)
    })

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
