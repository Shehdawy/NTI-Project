import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class InsufficientDataError(Exception):
    pass


def train_store_model(df: pd.DataFrame, date_col: str, sales_col: str,
                       promo_col: str = None, holiday_col: str = None,
                       store_col: str = None, output_dir: str = None) -> dict:
    """
    Trains a Random Forest sales-forecasting model on a store's own data and
    writes model.pkl, feature_config.json, business_insights.json, and
    history.csv into output_dir. Returns the same feature_config dict that
    was written to disk (includes real, computed evaluation metrics).

    Random Forest is used (rather than offering a model choice) because it
    performs well with no hyperparameter tuning across a wide range of retail
    datasets -- appropriate for a self-serve onboarding flow where the store
    owner isn't expected to be a data scientist.
    """
    df = df.copy()

    if date_col not in df.columns or sales_col not in df.columns:
        raise ValueError("The selected date/sales columns were not found in the uploaded file.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    df = df[df[sales_col] >= 0]
    df = df.sort_values(date_col)

    if len(df) < 30:
        raise InsufficientDataError(
            f"Only {len(df)} usable rows after cleaning -- need at least 30 rows "
            f"of historical sales data to train a reliable model."
        )

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Day"] = df[date_col].dt.day
    df["DayOfWeek"] = df[date_col].dt.dayofweek + 1
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    features = ["Year", "Month", "Week", "Day", "DayOfWeek", "IsWeekend"]

    has_promo = bool(promo_col and promo_col in df.columns)
    if has_promo:
        df["Promo"] = pd.to_numeric(df[promo_col], errors="coerce").fillna(0).astype(int)
        features.append("Promo")

    has_holiday = bool(holiday_col and holiday_col in df.columns)
    if has_holiday:
        df["Holiday"] = pd.to_numeric(df[holiday_col], errors="coerce").fillna(0).astype(int)
        features.append("Holiday")

    has_store_dimension = bool(store_col and store_col in df.columns)
    store_map = None
    if has_store_dimension:
        df["StoreName"] = df[store_col].astype(str)
        store_categories = sorted(df["StoreName"].unique())
        store_map = {name: i for i, name in enumerate(store_categories)}
        df["StoreCode"] = df["StoreName"].map(store_map)
        features.append("StoreCode")

    target = sales_col

    n = len(df)
    split_idx = int(n * 0.85)
    split_idx = min(split_idx, n - 1)  # always leave at least 1 row for validation
    split_idx = max(split_idx, 1)      # always leave at least 1 row for training
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, pred))
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = float(r2_score(y_valid, pred)) if len(y_valid) > 1 else None

    # ---- Generic, data-derived business insights (no invented numbers) ----
    insights = {}

    if has_promo:
        promo_avg = df.groupby("Promo")[target].mean()
        if 0 in promo_avg.index and 1 in promo_avg.index and promo_avg[0] > 0:
            insights["promotion"] = {
                "source": "PRIMARY DATA",
                "avg_no_promo": round(float(promo_avg[0]), 2),
                "avg_with_promo": round(float(promo_avg[1]), 2),
                "sales_lift_pct": round(float((promo_avg[1] / promo_avg[0] - 1) * 100), 2),
            }

    dow_avg = df.groupby("DayOfWeek")[target].mean().round(2).to_dict()
    month_avg = df.groupby("Month")[target].mean().round(2).to_dict()
    insights["seasonality"] = {
        "source": "PRIMARY DATA",
        "avg_by_day_of_week": {int(k): v for k, v in dow_avg.items()},
        "avg_by_month": {int(k): v for k, v in month_avg.items()},
        "best_day_of_week": int(max(dow_avg, key=dow_avg.get)),
        "best_month": int(max(month_avg, key=month_avg.get)),
    }

    insights["overview"] = {
        "source": "PRIMARY DATA",
        "rows_used": int(n),
        "date_range": [str(df[date_col].min().date()), str(df[date_col].max().date())],
        "avg_sales_all": round(float(df[target].mean()), 2),
    }

    if has_store_dimension:
        store_perf = df.groupby("StoreName")[target].mean().sort_values(ascending=False)
        insights["store_performance"] = {
            "source": "PRIMARY DATA",
            "top_5": store_perf.head(5).round(2).to_dict(),
            "bottom_5": store_perf.tail(5).round(2).to_dict(),
            "avg_across_stores": round(float(store_perf.mean()), 2),
        }

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))

    feature_config = {
        "features": features,
        "target": target,
        "date_col": date_col,
        "has_promo": has_promo,
        "has_holiday": has_holiday,
        "has_store_dimension": has_store_dimension,
        "store_map": store_map,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows_trained": int(len(train_df)),
        "rows_validated": int(len(valid_df)),
    }
    with open(os.path.join(output_dir, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=2)
    with open(os.path.join(output_dir, "business_insights.json"), "w") as f:
        json.dump(insights, f, indent=2, default=str)

    history_cols = [date_col, target] + (["StoreName"] if has_store_dimension else [])
    df[history_cols].to_csv(os.path.join(output_dir, "history.csv"), index=False)

    return feature_config
