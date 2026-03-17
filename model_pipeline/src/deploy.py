import json
import shutil
from pathlib import Path
from datetime import datetime

DECISION_PATH = Path(r"artifacts/reports/register_decision.json")
PROD_MODEL_PATH = Path(r"registry/production/model.joblib")
CURRENT_META_PATH = Path(r"registry/current_model.json")

def _copy_file_best_effort(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        shutil.copyfile(src, dst)

def main():
    if not DECISION_PATH.exists():
        raise FileNotFoundError(f"Missing decision file: {DECISION_PATH}")

    with open(DECISION_PATH, "r", encoding="utf-8") as f:
        decision = json.load(f)

    if not decision.get("promote", False):
        print({"status": "skipped", "reason": "candidate not better than production"})
        return

    src_model = Path(decision["candidate_model_path"])
    if not src_model.exists():
        raise FileNotFoundError(f"Missing candidate model: {src_model}")

    _copy_file_best_effort(src_model, PROD_MODEL_PATH)

    current_meta = {
        "model_name": "churn_prediction_model",
        "model_path": str(PROD_MODEL_PATH).replace("\\", "/"),
        "updated_at": datetime.now().isoformat(),
        "metrics": decision["metrics"]
    }

    with open(CURRENT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(current_meta, f, indent=2)

    print({"status": "deployed", "production_model": str(PROD_MODEL_PATH)})

if __name__ == "__main__":
    main()
