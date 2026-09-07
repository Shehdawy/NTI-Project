"""Load and expose the existing trained model and metadata."""

import json
from pathlib import Path
from typing import Any

import joblib


class ModelService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_path = root / "models" / "demand_model.pkl"
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def load(self) -> None:
        self.model = joblib.load(self.model_path)
        metadata_path = self.root / "models" / "metadata.json"
        metrics_path = self.root / "models" / "model_metrics.csv"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metrics_path.exists():
            import pandas as pd
            metrics = pd.read_csv(metrics_path)
            best = metrics[metrics["model"] == self.metadata.get("best_model")]
            if not best.empty:
                row = best.iloc[0]
                self.metrics = {key: float(row[key]) for key in ("MAE", "RMSE", "R2")}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, frame):
        if self.model is None:
            raise RuntimeError("The demand model is not loaded.")
        return self.model.predict(frame)