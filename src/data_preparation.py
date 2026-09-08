"""M5 dataset loading and feature preparation for demand modeling."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_uploaded_dataset(
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    sample_every: int = 50,
    recent_days: int = 180,
) -> pd.DataFrame:
    """Load the uploaded M5 files into the tabular format used by the project."""
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    recent_calendar = calendar.tail(recent_days).copy()
    identity_columns = ["id", "item_id", "cat_id", "store_id", "state_id"]
    sales_columns = pd.read_csv(sales_path, nrows=0).columns
    day_columns = [day for day in recent_calendar["d"] if day in sales_columns]
    recent_calendar = recent_calendar[recent_calendar["d"].isin(day_columns)]
    sales = pd.read_csv(
        sales_path,
        usecols=identity_columns + day_columns,
        skiprows=lambda index: index > 0 and index % sample_every != 1,
    )
    long_sales = sales.melt(
        id_vars=identity_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="units_sold",
    )
    long_sales = long_sales.merge(recent_calendar, on="d", how="left")
    item_prices = []
    selected_items = set(sales["item_id"])
    selected_stores = set(sales["store_id"])
    for chunk in pd.read_csv(prices_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"], chunksize=500_000):
        filtered = chunk[chunk["item_id"].isin(selected_items) & chunk["store_id"].isin(selected_stores)]
        if not filtered.empty:
            item_prices.append(filtered)
    prices = pd.concat(item_prices, ignore_index=True)
    long_sales = long_sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_sales["selling_price"] = long_sales["sell_price"]
    long_sales["product_category"] = long_sales["cat_id"]
    long_sales["cost_price"] = long_sales["selling_price"] * 0.65
    long_sales["competitor_price"] = long_sales["selling_price"] * 1.03
    long_sales["inventory"] = 100
    long_sales["promotion"] = (
        long_sales["event_type_1"].notna()
        | long_sales["event_type_2"].notna()
        | long_sales[["snap_CA", "snap_TX", "snap_WI"]].eq(1).any(axis=1)
    ).astype(int)
    long_sales["season"] = np.select(
        [long_sales["month"] <= 2, long_sales["month"] <= 5, long_sales["month"] <= 8],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    long_sales["day_of_week"] = long_sales["weekday"]
    long_sales["customer_rating"] = 4.1
    long_sales["demand"] = long_sales["units_sold"]
    long_sales["revenue"] = long_sales["units_sold"] * long_sales["selling_price"]
    long_sales["profit"] = long_sales["units_sold"] * (long_sales["selling_price"] - long_sales["cost_price"])
    return long_sales[[
        "item_id", "product_category", "date", "cost_price", "selling_price",
        "competitor_price", "units_sold", "inventory", "promotion", "season",
        "day_of_week", "customer_rating", "demand", "revenue", "profit",
    ]].rename(columns={"item_id": "product_id"})


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features available when a price decision is made."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["month"] = result["date"].dt.month
    result["weekend"] = (result["date"].dt.dayofweek >= 5).astype(int)
    result["promotion_flag"] = result["promotion"].astype(int)
    result["price_difference_from_competitor"] = result["selling_price"] - result["competitor_price"]
    result["price_ratio_to_competitor"] = result["selling_price"] / result["competitor_price"].clip(lower=1)
    return result


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning while preserving valid business observations."""
    result = frame.copy()
    result = result.drop_duplicates().reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    numeric_columns = [
        "cost_price", "selling_price", "competitor_price", "units_sold",
        "inventory", "promotion", "customer_rating", "demand", "revenue", "profit",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    required_columns = [
        "product_id", "product_category", "date", "season", "selling_price",
        "units_sold", "promotion", "customer_rating",
    ]
    result = result.dropna(subset=required_columns)
    positive_columns = ["cost_price", "selling_price", "competitor_price", "inventory"]
    for column in positive_columns:
        result = result[result[column] > 0]
    result["promotion"] = result["promotion"].clip(0, 1).round().astype(int)
    result["customer_rating"] = result["customer_rating"].clip(1, 5)
    return result.reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)