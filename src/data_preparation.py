"""M5 dataset loading and feature preparation for demand modeling."""

from pathlib import Path

import numpy as np
import pandas as pd

# Rough business assumptions used because the M5 dataset only tells us
# what actually sold, not what it cost or what competitors charged.
# These are clearly labelled here and in data/README.md -- they are NOT
# real cost or competitor data.
ASSUMED_COST_MARGIN = 0.65  # cost is assumed to be 65% of the selling price
ASSUMED_COMPETITOR_MARKUP = 1.03  # competitor price is assumed to be 3% above ours
ASSUMED_INVENTORY_UNITS = 100
ASSUMED_CUSTOMER_RATING = 4.1

OUTPUT_COLUMNS = [
    "item_id", "product_category", "date", "cost_price", "selling_price",
    "competitor_price", "units_sold", "inventory", "promotion", "season",
    "day_of_week", "customer_rating", "demand", "revenue", "profit",
]


def load_uploaded_dataset(
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    sample_every: int = 50,
    recent_days: int = 180,
) -> pd.DataFrame:
    """Load the uploaded M5 files into the tabular format used by the project.

    The M5 files are large (millions of rows), so this samples one row in
    every ``sample_every`` products and keeps only the most ``recent_days``
    days, which keeps the project runnable on a laptop without changing the
    shape of the resulting table.
    """
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

    # M5 stores one column per day; reshape to one row per (item, day).
    long_sales = sales.melt(
        id_vars=identity_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="units_sold",
    )
    long_sales = long_sales.merge(recent_calendar, on="d", how="left")

    # Prices live in a separate, even larger file -- read it in chunks and
    # keep only rows for the items/stores we actually sampled above.
    selected_items = set(sales["item_id"])
    selected_stores = set(sales["store_id"])
    price_chunks = []
    for chunk in pd.read_csv(prices_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"], chunksize=500_000):
        filtered = chunk[chunk["item_id"].isin(selected_items) & chunk["store_id"].isin(selected_stores)]
        if not filtered.empty:
            price_chunks.append(filtered)
    prices = pd.concat(price_chunks, ignore_index=True)
    long_sales = long_sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long_sales["selling_price"] = long_sales["sell_price"]
    long_sales["product_category"] = long_sales["cat_id"]

    # --- Assumed business fields (M5 doesn't provide these) ---
    long_sales["cost_price"] = long_sales["selling_price"] * ASSUMED_COST_MARGIN
    long_sales["competitor_price"] = long_sales["selling_price"] * ASSUMED_COMPETITOR_MARKUP
    long_sales["inventory"] = ASSUMED_INVENTORY_UNITS
    long_sales["customer_rating"] = ASSUMED_CUSTOMER_RATING

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

    long_sales["demand"] = long_sales["units_sold"]
    long_sales["revenue"] = long_sales["units_sold"] * long_sales["selling_price"]
    long_sales["profit"] = long_sales["units_sold"] * (long_sales["selling_price"] - long_sales["cost_price"])

    return long_sales[OUTPUT_COLUMNS].rename(columns={"item_id": "product_id"})


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features available at the moment a price decision is made."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["month"] = result["date"].dt.month
    result["weekend"] = (result["date"].dt.dayofweek >= 5).astype(int)
    result["promotion_flag"] = result["promotion"].astype(int)
    result["price_difference_from_competitor"] = result["selling_price"] - result["competitor_price"]
    result["price_ratio_to_competitor"] = result["selling_price"] / result["competitor_price"].clip(lower=1)
    return result


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning while preserving valid business observations.

    "Conservative" here means: drop rows that are structurally broken
    (missing required fields, non-positive prices), but don't try to be
    clever about outliers -- a very high or low sale is still real data.
    """
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
    """Write the prepared dataset to disk, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
