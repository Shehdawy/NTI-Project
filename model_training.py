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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.deep_learning_model import train_deep_learning
from src.data_preparation import add_features, clean_dataset, load_uploaded_dataset, save_dataset

ROOT = Path(__file__).parent
SALES_PATH = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
PRICES_PATH = ROOT / "data" / "raw" / "sell_prices.csv"
CALENDAR_PATH = ROOT / "data" / "raw" / "calendar.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
PREPARED_DATA_PATH = ROOT / "data" / "processed" / "prepared_sales_data.csv"
FEATURES = [
    "product_category", "selling_price", "promotion", "season", "day_of_week",
    "month", "weekend",
]


def build_pipeline(model):
    categorical = ["product_category", "season", "day_of_week"]
    numerical = ["selling_price", "promotion", "month", "weekend"]
    transformer = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", StandardScaler(), numerical)])
    return Pipeline([("features", transformer), ("model", model)])


def create_eda(frame: pd.DataFrame) -> None:
    output = ROOT / "reports" / "eda_overview.png"
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
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    raw = load_uploaded_dataset(SALES_PATH, PRICES_PATH, CALENDAR_PATH)
    source = "uploaded M5 sales, price, and calendar files"
    raw = clean_dataset(raw)
    save_dataset(raw, PREPARED_DATA_PATH)
    report = {
        "dataset": [path.name for path in (SALES_PATH, PRICES_PATH, CALENDAR_PATH)],
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
        "source_fields": ["product_category", "date", "selling_price", "units_sold", "promotion", "season", "day_of_week"],
        "derived_assumption_fields": ["cost_price", "competitor_price", "inventory", "customer_rating", "revenue", "profit"],
    }
    with open(ROOT / "models" / "data_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    frame = add_features(raw)
    create_eda(frame)
    features = FEATURES
    frame = frame.sort_values("date").reset_index(drop=True)
    unique_dates = frame["date"].drop_duplicates().sort_values().to_numpy()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_frame = frame[frame["date"] < split_date]
    test_frame = frame[frame["date"] >= split_date]
    X_train, X_test = train_frame[features], test_frame[features]
    y_train, y_test = train_frame["units_sold"], test_frame["units_sold"]
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
    joblib.dump(best_model, MODEL_PATH, compress=3)
    metrics.to_csv(ROOT / "models" / "model_metrics.csv", index=False)
    prediction_output = X_test.copy()
    prediction_output["actual_demand"] = y_test.to_numpy()
    prediction_output["predicted_demand"] = best_model.predict(X_test)
    prediction_output.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)
    transformer = best_model.named_steps["features"]
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": transformer.get_feature_names_out(), "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).head(15)
        importance.to_csv(ROOT / "models" / "feature_importance.csv", index=False)
    with open(ROOT / "models" / "metadata.json", "w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "rows": len(frame), "features": features, "data_start": str(frame["date"].min()), "data_end": str(frame["date"].max()), "split": "chronological 80/20 by date", "split_date": str(pd.Timestamp(split_date))}, file, indent=2)
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