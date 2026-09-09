"""Load and expose the existing trained model and its saved metadata/metrics."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelService:
    """Holds the one trained demand model the whole API shares.

    Loaded once at startup (see ``backend.main.lifespan``) rather than per
    request, since re-reading a joblib file on every call would be slow and
    unnecessary -- the model doesn't change while the server is running.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_path = root / "models" / "demand_model.joblib"
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def load(self) -> None:
        """Load the model file and its accompanying metadata/metrics, if present."""
        self.model = joblib.load(self.model_path)

        metadata_path = self.root / "models" / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        metrics_path = self.root / "models" / "model_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            best_row = metrics[metrics["model"] == self.metadata.get("best_model")]
            if not best_row.empty:
                row = best_row.iloc[0]
                self.metrics = {key: float(row[key]) for key in ("MAE", "RMSE", "R2")}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, frame: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("The demand model is not loaded.")
        return self.model.predict(frame)
