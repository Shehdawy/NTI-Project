"""Demand feature preparation shared by the API's prediction and pricing calls."""

import pandas as pd


def build_input(request, price: float | None = None) -> pd.DataFrame:
    """Turn one API request into the single-row feature frame the model expects.

    ``price`` lets callers override the request's own price -- used by the
    pricing engine, which needs to ask "what if the price were X?" for many
    candidate prices without constructing a new request object each time.
    """
    decision_date = pd.Timestamp(request.date)
    selling_price = request.price if price is None else price
    return pd.DataFrame([{
        "product_category": request.product_category,
        "cost_price": request.cost_price,
        "selling_price": selling_price,
        "competitor_price": request.competitor_price,
        "inventory": request.inventory,
        "promotion": int(request.promotion),
        "season": request.season,
        "day_of_week": decision_date.day_name(),
        "customer_rating": request.customer_rating,
        "month": decision_date.month,
        "weekend": int(decision_date.dayofweek >= 5),
        "promotion_flag": int(request.promotion),
        "price_difference_from_competitor": selling_price - request.competitor_price,
        "price_ratio_to_competitor": selling_price / request.competitor_price,
    }])


def predict_demand(model_service, request) -> float:
    """Predict demand for one request, clipped to be non-negative."""
    return max(0.0, float(model_service.predict(build_input(request))[0]))
