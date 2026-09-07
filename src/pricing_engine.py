"""Business-facing price simulation logic."""

import numpy as np
import pandas as pd


def recommend_price(
    model,
    category: str,
    cost_price: float | None,
    current_price: float,
    competitor_price: float,
    inventory: int,
    promotion: int,
    season: str,
    rating: float = 4.1,
    objective: str = "Maximize profit",
    decision_date: str = "2024-12-31",
) -> tuple[dict, pd.DataFrame]:
    """Simulate candidate prices and select one according to a business objective."""
    if cost_price is not None and cost_price <= 0:
        raise ValueError("Cost must be positive when supplied.")
    if current_price <= 0 or competitor_price <= 0 or inventory <= 0:
        raise ValueError("Prices and inventory must be positive.")
    parsed_date = pd.Timestamp(decision_date)
    lower = current_price * 0.80
    if cost_price is not None:
        lower = max(cost_price * 1.08, lower)
    upper = min(competitor_price * 1.20, current_price * 1.20)
    candidates = np.unique(np.round(np.linspace(lower, max(lower, upper), 25), 2))
    rows = []
    for price in candidates:
        input_frame = pd.DataFrame([{
            "product_category": category,
            "cost_price": cost_price,
            "selling_price": price,
            "competitor_price": competitor_price,
            "inventory": inventory,
            "promotion": promotion,
            "season": season,
            "day_of_week": parsed_date.day_name(),
            "customer_rating": rating,
            "month": parsed_date.month,
            "weekend": int(parsed_date.dayofweek >= 5),
            "promotion_flag": promotion,
            "price_difference_from_competitor": price - competitor_price,
            "price_ratio_to_competitor": price / max(competitor_price, 1),
        }])
        demand = float(np.maximum(model.predict(input_frame)[0], 0))
        demand = min(demand, float(inventory))
        revenue = demand * price
        profit = demand * (price - cost_price) if cost_price is not None else None
        balanced_score = revenue if profit is None else revenue * 0.5 + profit * 0.5
        rows.append({"candidate_price": price, "predicted_demand": demand, "expected_revenue": revenue, "expected_profit": profit, "profit_margin": profit / revenue if revenue else 0, "balanced_score": balanced_score})
    simulation = pd.DataFrame(rows)
    objective_columns = {"Maximize revenue": "expected_revenue", "Maximize profit": "expected_profit", "Balanced": "balanced_score"}
    if objective not in objective_columns:
        raise ValueError("Choose a supported recommendation objective.")
    best = simulation.sort_values(objective_columns[objective], ascending=False).iloc[0]
    expected_profit = best["expected_profit"]
    result = {"recommended_price": float(best["candidate_price"]), "predicted_demand": float(best["predicted_demand"]), "expected_revenue": float(best["expected_revenue"]), "expected_profit": float(expected_profit) if pd.notna(expected_profit) else None, "profit_margin": float(best["profit_margin"]) if pd.notna(expected_profit) else None}
    return result, simulation