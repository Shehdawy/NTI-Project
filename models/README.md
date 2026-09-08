# Models

Generated model files are ignored by Git because trained artifacts can be large and environment-specific.

Run `python model_training.py` after placing the M5 files in `data/raw/`. The training script creates `demand_model.joblib`, `metadata.json`, `model_metrics.csv`, and feature-importance output here.

The application loads `models/demand_model.joblib`.
