from pathlib import Path
import shutil
from datetime import datetime

SOURCE = Path(r"data-pipeline/data/Newdata/customer_churn.csv")
RAW_DIR = Path(r"data-pipeline/data/raw")

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = RAW_DIR / f"customer_churn_{run_id}.csv"
    latest = RAW_DIR / "latest.csv"

    shutil.copyfile(SOURCE, target)
    shutil.copyfile(SOURCE, latest)

    print({
        "raw_path": str(target),
        "latest_path": str(latest),
        "run_id": run_id
    })

if __name__ == "__main__":
    main()