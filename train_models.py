"""Generate data, train demand models, and save dashboard artifacts."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dl_model import train_deep_learning
from src.preprocessing import add_features, clean_dataset, generate_dataset, save_dataset

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "sales_data.csv"
MODEL_PATH = ROOT / "models" / "demand_model.pkl"


def build_pipeline(model):
    categorical = ["product_category", "season", "day_of_week"]
    numerical = ["cost_price", "selling_price", "competitor_price", "inventory", "promotion", "customer_rating", "month", "weekend", "promotion_flag", "price_difference_from_competitor", "price_ratio_to_competitor"]
    transformer = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", StandardScaler(), numerical)])
    return Pipeline([("features", transformer), ("model", model)])


def create_eda(frame: pd.DataFrame) -> None:
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(frame["selling_price"], ax=axes[0, 0], color="#0f766e")
    axes[0, 0].set_title("Selling price distribution")
    sns.histplot(frame["units_sold"], ax=axes[0, 1], color="#f97316")
    axes[0, 1].set_title("Demand distribution")
    sns.scatterplot(data=frame.sample(min(1500, len(frame)), random_state=42), x="selling_price", y="units_sold", hue="promotion", alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Price and demand")
    sns.barplot(data=frame, x="product_category", y="units_sold", ax=axes[1, 1], color="#2563eb")
    axes[1, 1].set_title("Average demand by category")
    fig.tight_layout()
    fig.savefig(output / "eda_overview.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        source = "supplied data/processed/sales_data.csv"
    else:
        raw = generate_dataset()
        source = "generated fallback dataset"
    raw = clean_dataset(raw)
    save_dataset(raw, DATA_PATH)
    report = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "column_names": list(raw.columns),
        "data_types": {column: str(value) for column, value in raw.dtypes.items()},
        "missing_values": {column: int(value) for column, value in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "date_range": {"start": str(raw["date"].min()), "end": str(raw["date"].max())},
        "numeric_columns": list(raw.select_dtypes(include="number").columns),
        "categorical_columns": list(raw.select_dtypes(exclude="number").columns),
        "target": "units_sold",
        "real_pricing_fields": ["selling_price", "cost_price", "competitor_price", "revenue", "profit"],
    }
    with open(ROOT / "models" / "data_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    frame = add_features(raw)
    create_eda(frame)
    features = ["product_category", "cost_price", "selling_price", "competitor_price", "inventory", "promotion", "season", "day_of_week", "customer_rating", "month", "weekend", "promotion_flag", "price_difference_from_competitor", "price_ratio_to_competitor"]
    X = frame[features]
    y = frame["units_sold"]
    frame = frame.sort_values("date").reset_index(drop=True)
    X = frame[features]
    y = frame["units_sold"]
    split = int(len(frame) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    baseline_prediction = [float(y_train.mean())] * len(y_test)
    candidates = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1), "Gradient Boosting": GradientBoostingRegressor(n_estimators=160, learning_rate=0.05, max_depth=3, random_state=42)}
    results = [{"model": "Mean baseline", "MAE": mean_absolute_error(y_test, baseline_prediction), "RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5, "R2": r2_score(y_test, baseline_prediction)}]
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        results.append({"model": name, "MAE": mean_absolute_error(y_test, prediction), "RMSE": mean_squared_error(y_test, prediction) ** 0.5, "R2": r2_score(y_test, prediction)})
        fitted[name] = pipeline
    metrics = pd.DataFrame(results).sort_values("RMSE")
    best_name = metrics[metrics["model"] != "Mean baseline"].iloc[0]["model"]
    best_model = fitted[best_name]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metrics.to_csv(ROOT / "models" / "model_metrics.csv", index=False)
    prediction_output = X_test.copy()
    prediction_output["actual_demand"] = y_test.to_numpy()
    prediction_output["predicted_demand"] = best_model.predict(X_test)
    prediction_output.to_csv(ROOT / "models" / "test_predictions.csv", index=False)
    transformer = best_model.named_steps["features"]
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": transformer.get_feature_names_out(), "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).head(15)
        importance.to_csv(ROOT / "models" / "feature_importance.csv", index=False)
    with open(ROOT / "models" / "metadata.json", "w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "rows": len(frame), "features": features, "data_start": str(frame["date"].min()), "data_end": str(frame["date"].max()), "split": "chronological 80/20"}, file, indent=2)
    encoded_train = pd.get_dummies(X_train, columns=["product_category", "season", "day_of_week"])
    encoded_test = pd.get_dummies(X_test, columns=["product_category", "season", "day_of_week"]).reindex(columns=encoded_train.columns, fill_value=0)
    dl_result = train_deep_learning(encoded_train, y_train, encoded_test, y_test, ROOT / "models" / "dl_model.keras")
    with open(ROOT / "models" / "dl_results.json", "w", encoding="utf-8") as file:
        json.dump(dl_result, file, indent=2)
    print(f"Training source: {source}")
    print(f"Rows after cleaning: {len(raw)}")
    print(metrics.to_string(index=False))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Deep learning: {dl_result['status']}")


if __name__ == "__main__":
    main()