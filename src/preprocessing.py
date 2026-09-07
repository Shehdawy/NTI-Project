"""Synthetic data generation and feature preparation for demand modeling."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
CATEGORIES = ["Electronics", "Home", "Beauty", "Grocery", "Sports"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]


def generate_dataset(rows: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a reproducible sales dataset with business-shaped relationships."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=rows)
    category = rng.choice(CATEGORIES, rows, p=[0.22, 0.22, 0.18, 0.23, 0.15])
    category_cost = {"Electronics": 210, "Home": 95, "Beauty": 55, "Grocery": 24, "Sports": 80}
    category_demand = {"Electronics": 12, "Home": 18, "Beauty": 22, "Grocery": 35, "Sports": 20}
    cost = np.array([category_cost[item] for item in category]) * rng.normal(1, 0.16, rows)
    cost = np.maximum(cost, 5).round(2)
    competitor = (cost * rng.uniform(1.25, 1.8, rows)).round(2)
    selling = (competitor * rng.uniform(0.88, 1.12, rows)).round(2)
    inventory = rng.integers(15, 260, rows)
    promotion = rng.binomial(1, 0.25, rows)
    rating = np.clip(rng.normal(4.1, 0.45, rows), 2.5, 5).round(2)
    month = dates.month.to_numpy()
    season = np.select([month <= 2, month <= 5, month <= 8], ["Winter", "Spring", "Summer"], default="Autumn")
    weekend = (dates.dayofweek.to_numpy() >= 5).astype(int)
    seasonal_effect = np.select([season == "Summer", season == "Winter", season == "Autumn"], [4, 2, 1], default=0)
    price_pressure = (selling / competitor - 1) * 24
    demand_mean = (
        np.array([category_demand[item] for item in category])
        + seasonal_effect
        + promotion * 11
        + (rating - 4) * 5
        + weekend * 2
        - price_pressure
        + (competitor - selling) / np.maximum(cost, 1) * 3
        + np.minimum(inventory, 100) / 100 * 2
    )
    units_sold = rng.poisson(np.maximum(demand_mean, 2)).astype(int)
    units_sold = np.minimum(units_sold, inventory)
    revenue = (units_sold * selling).round(2)
    profit = (units_sold * (selling - cost)).round(2)
    frame = pd.DataFrame({
        "product_id": [f"P-{number:04d}" for number in rng.integers(1, 101, rows)],
        "product_category": category,
        "date": dates,
        "cost_price": cost,
        "selling_price": selling,
        "competitor_price": competitor,
        "units_sold": units_sold,
        "inventory": inventory,
        "promotion": promotion,
        "season": season,
        "day_of_week": dates.day_name(),
        "customer_rating": rating,
        "demand": units_sold,
        "revenue": revenue,
        "profit": profit,
    })
    return frame


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
        result[column] = result[column].fillna(result[column].median())
    result = result.dropna(subset=["product_id", "product_category", "date", "season"])
    positive_columns = ["cost_price", "selling_price", "competitor_price", "inventory"]
    for column in positive_columns:
        result = result[result[column] > 0]
    result["promotion"] = result["promotion"].clip(0, 1).round().astype(int)
    result["customer_rating"] = result["customer_rating"].clip(1, 5)
    return result.reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)