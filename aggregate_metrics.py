import datetime
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
PRED_DIR = ROOT_DIR / "predictions"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate_metrics() -> dict:
    metrics_files = sorted(PRED_DIR.glob("*_metrics.json"))
    per_model = {}
    runs = 0

    for metrics_path in metrics_files:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", {})
        if not isinstance(models, dict):
            continue
        runs += 1

        for model_name, stats in models.items():
            if not isinstance(stats, dict):
                continue
            entry = per_model.setdefault(
                model_name,
                {
                    "correct": 0,
                    "total": 0,
                    "char_edits": 0,
                    "char_total": 0,
                    "cer_weighted_sum": 0.0,
                    "cer_weighted_total": 0.0,
                    "runs": 0,
                },
            )

            total = int(stats.get("total", 0) or 0)
            correct = int(stats.get("correct", 0) or 0)
            entry["correct"] += correct
            entry["total"] += total
            entry["runs"] += 1

            char_edits = stats.get("char_edits")
            char_total = stats.get("char_total")
            if char_edits is not None and char_total is not None:
                entry["char_edits"] += int(char_edits)
                entry["char_total"] += int(char_total)
            elif total:
                cer_value = _safe_float(stats.get("cer"))
                entry["cer_weighted_sum"] += cer_value * total
                entry["cer_weighted_total"] += total

    summary_models = {}
    for model_name, stats in per_model.items():
        total = stats["total"]
        accuracy = (stats["correct"] / total) if total else 0.0
        if stats["char_total"]:
            cer = stats["char_edits"] / stats["char_total"]
            cer_method = "exact"
        elif stats["cer_weighted_total"]:
            cer = stats["cer_weighted_sum"] / stats["cer_weighted_total"]
            cer_method = "weighted_by_total"
        else:
            cer = 0.0
            cer_method = "unavailable"

        char_accuracy = (1.0 - cer) if cer_method != "unavailable" else 0.0

        summary_models[model_name] = {
            "exact_match_accuracy": round(accuracy,5),
            "character_level_accuracy": round(char_accuracy,5),
            "character_error_rate (CER)": round(cer,5)
        }

    return {
        "generated_at": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "models": summary_models,
    }


def main() -> None:
    summary = aggregate_metrics()
    out_path = PRED_DIR / "metrics_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary metrics to {out_path}")


if __name__ == "__main__":
    main()
