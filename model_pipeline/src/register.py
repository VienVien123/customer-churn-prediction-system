import json
from pathlib import Path

CURRENT_MODEL_META = Path(r"registry/current_model.json")
NEW_METRICS_PATH = Path(r"artifacts/metrics/eval_latest.json")

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    new_metrics = load_json(NEW_METRICS_PATH)
    if new_metrics is None:
        raise FileNotFoundError(f"Missing metrics file: {NEW_METRICS_PATH}")

    current_meta = load_json(CURRENT_MODEL_META)

    promote = False

    if current_meta is None:
        promote = True
    else:
        current_f1 = current_meta.get("metrics", {}).get("f1_score", 0.0)
        new_f1 = new_metrics.get("f1_score", 0.0)
        if new_f1 > current_f1:
            promote = True

    result = {
        "promote": promote,
        "metrics": new_metrics,
        "candidate_model_path": "artifacts/models/model.joblib"
    }

    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    with open("artifacts/reports/register_decision.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(result)

if __name__ == "__main__":
    main()
