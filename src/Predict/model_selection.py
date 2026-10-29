"""Persisted XGBoost model selection based on comparable held-out results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
REGISTRY_NAME = "selected_models.json"
HISTORY_NAME = "model_candidates.json"


def _path(model_dir, name):
    return Path(model_dir) / name


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def selected_model_path(kind, model_dir=MODEL_DIR):
    """Return the promoted model, falling back to legacy discovery when needed."""
    model_dir = Path(model_dir)
    selected = _read(_path(model_dir, REGISTRY_NAME), {}).get(kind, {})
    path = model_dir / selected.get("filename", "")
    if path.exists():
        return path
    candidates = [path for path in model_dir.glob(f"*{kind}*.json")
                  if not path.stem.endswith("_features") and (model_dir / f"{path.stem}_features.json").exists()]
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime) if candidates else None


def _evaluation_key(experiment):
    """Identity of the fixed data/split protocol used to compare candidates."""
    if not experiment:
        return None
    dataset = experiment.get("dataset", {})
    split = experiment.get("final_split", {})
    if not dataset.get("sha256") or not split.get("final_start"):
        return None
    return {
        "dataset_sha256": dataset["sha256"],
        "final_start": split["final_start"],
        "final_end": split.get("final_end"),
        "calibration": experiment.get("calibration"),
    }


def record_candidate(kind, model_path, dataset, accuracy, log_loss, model_dir=MODEL_DIR,
                     experiment=None):
    """Record a candidate and compare it only with the same data/split protocol."""
    model_dir = Path(model_dir)
    registry_path = _path(model_dir, REGISTRY_NAME)
    history_path = _path(model_dir, HISTORY_NAME)
    registry = _read(registry_path, {})
    history = _read(history_path, [])
    candidate = {
        "kind": kind,
        "filename": Path(model_path).name,
        "dataset": dataset,
        "test_accuracy": float(accuracy),
        "test_log_loss": float(log_loss),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    evaluation_key = _evaluation_key(experiment)
    if evaluation_key:
        candidate["evaluation_key"] = evaluation_key
        candidate["experiment_report"] = f"{Path(model_path).stem}_experiment.json"
    incumbent = registry.get(kind)
    comparable = incumbent and incumbent.get("dataset") == dataset
    if evaluation_key and incumbent:
        # A legacy registry entry is allowed to seed the first audited cycle.
        # Once an audited entry exists, a changed data fingerprint, boundary, or
        # calibration method must not silently replace it.
        comparable = comparable and (
            "evaluation_key" not in incumbent or incumbent.get("evaluation_key") == evaluation_key
        )
    promoted = not incumbent or (comparable and candidate["test_log_loss"] < incumbent["test_log_loss"])
    candidate["promoted"] = promoted
    history.append(candidate)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    if promoted:
        registry[kind] = candidate
        registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    return promoted, incumbent
