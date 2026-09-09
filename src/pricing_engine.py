"""Business-facing price simulation logic.

The core idea (and the reason this is more than "predict one perfect
price"): given a business scenario, we build a handful of *candidate*
prices, ask the trained demand model how much would sell at each one, turn
that into expected revenue and profit, and then pick the candidate that
best satisfies the chosen business objective. Nothing here guesses the
"right" price directly -- it always compares real simulated alternatives.
"""

import numpy as np
import pandas as pd

# How far around the current price we're willing to search for a better one.
CANDIDATE_COUNT = 25
MAX_DISCOUNT_FROM_CURRENT = 0.80  # never go more than 20% below the current price
MAX_MARKUP_FROM_CURRENT = 1.20  # never go more than 20% above the current price
MAX_MARKUP_OVER_COMPETITOR = 1.20  # never go more than 20% above the competitor
MIN_MARKUP_OVER_COST = 1.08  # always keep at least an 8% margin over cost, if cost is known

OBJECTIVE_TO_SCORE_COLUMN = {
    "Maximize revenue": "expected_revenue",
    "Maximize profit": "expected_profit",
    "Balanced": "balanced_score",
}


def _validate_inputs(cost_price: float | None, current_price: float, competitor_price: float, inventory: int) -> None:
    """Raise ValueError for scenarios that can't produce a sensible recommendation."""
    if cost_price is not None and cost_price <= 0:
        raise ValueError("Cost must be positive when supplied.")
    if current_price <= 0 or competitor_price <= 0 or inventory <= 0:
        raise ValueError("Prices and inventory must be positive.")


def _candidate_price_range(cost_price: float | None, current_price: float, competitor_price: float) -> tuple[float, float]:
    """Compute the [lower, upper] price band worth testing for this scenario."""
    lower = current_price * MAX_DISCOUNT_FROM_CURRENT
    if cost_price is not None:
        # Never recommend a price so low it barely covers cost.
        lower = max(cost_price * MIN_MARKUP_OVER_COST, lower)
    upper = min(competitor_price * MAX_MARKUP_OVER_COMPETITOR, current_price * MAX_MARKUP_FROM_CURRENT)
    if lower > upper:
        raise ValueError("No feasible candidate price satisfies the cost, current-price, and competitor constraints.")
    return lower, upper


def _build_model_input(
    price: float,
    category: str,
    cost_price: float | None,
    competitor_price: float,
    inventory: int,
    promotion: int,
    season: str,
    rating: float,
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """Assemble a single-row feature frame for one candidate price."""
    return pd.DataFrame([{
        "product_category": category,
        "cost_price": cost_price,
        "selling_price": price,
        "competitor_price": competitor_price,
        "inventory": inventory,
        "promotion": promotion,
        "season": season,
        "day_of_week": decision_date.day_name(),
        "customer_rating": rating,
        "month": decision_date.month,
        "weekend": int(decision_date.dayofweek >= 5),
        "promotion_flag": promotion,
        "price_difference_from_competitor": price - competitor_price,
        "price_ratio_to_competitor": price / max(competitor_price, 1),
    }])


def _simulate_candidates(
    model,
    candidates: np.ndarray,
    category: str,
    cost_price: float | None,
    competitor_price: float,
    inventory: int,
    promotion: int,
    season: str,
    rating: float,
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """Predict demand, revenue, and profit for every candidate price."""
    rows = []
    for price in candidates:
        model_input = _build_model_input(
            price, category, cost_price, competitor_price, inventory, promotion, season, rating, decision_date,
        )
        predicted_demand = float(np.maximum(model.predict(model_input)[0], 0))
        predicted_demand = min(predicted_demand, float(inventory))  # can't sell more than we have

        revenue = predicted_demand * price
        profit = predicted_demand * (price - cost_price) if cost_price is not None else None
        profit_margin = profit / revenue if profit is not None and revenue else None

        rows.append({
            "candidate_price": price,
            "predicted_demand": predicted_demand,
            "expected_revenue": revenue,
            "expected_profit": profit,
            "profit_margin": profit_margin,
        })
    return pd.DataFrame(rows)


def _add_balanced_score(simulation: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-1 'balanced_score' column that weights revenue and profit equally.

    Falls back to plain revenue when profit isn't available (no cost price
    was supplied), since profit can't be balanced against something unknown.
    """
    if simulation["expected_profit"].notna().any():
        revenue_span = simulation["expected_revenue"].max() - simulation["expected_revenue"].min()
        profit_span = simulation["expected_profit"].max() - simulation["expected_profit"].min()
        revenue_score = (simulation["expected_revenue"] - simulation["expected_revenue"].min()) / revenue_span if revenue_span else 0.5
        profit_score = (simulation["expected_profit"] - simulation["expected_profit"].min()) / profit_span if profit_span else 0.5
        simulation["balanced_score"] = (revenue_score + profit_score) * 0.5
    else:
        simulation["balanced_score"] = simulation["expected_revenue"]
    return simulation


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
    """Simulate candidate prices and select one according to a business objective.

    Returns a tuple of:
    - ``result``: a dict describing the single recommended price and its
      predicted outcome (demand, revenue, profit, margin).
    - ``simulation``: a DataFrame with one row per candidate price tested,
      useful for charting the full trade-off curve.
    """
    _validate_inputs(cost_price, current_price, competitor_price, inventory)
    if objective not in OBJECTIVE_TO_SCORE_COLUMN:
        raise ValueError("Choose a supported recommendation objective.")

    lower, upper = _candidate_price_range(cost_price, current_price, competitor_price)
    candidates = np.unique(np.round(np.linspace(lower, upper, CANDIDATE_COUNT), 2))
    parsed_date = pd.Timestamp(decision_date)

    simulation = _simulate_candidates(
        model, candidates, category, cost_price, competitor_price, inventory, promotion, season, rating, parsed_date,
    )
    simulation = _add_balanced_score(simulation)

    score_column = OBJECTIVE_TO_SCORE_COLUMN[objective]
    best = simulation.sort_values(score_column, ascending=False).iloc[0]
    expected_profit = best["expected_profit"]

    result = {
        "recommended_price": float(best["candidate_price"]),
        "predicted_demand": float(best["predicted_demand"]),
        "expected_revenue": float(best["expected_revenue"]),
        "expected_profit": float(expected_profit) if pd.notna(expected_profit) else None,
        "profit_margin": float(best["profit_margin"]) if pd.notna(expected_profit) else None,
    }
    return result, simulation
