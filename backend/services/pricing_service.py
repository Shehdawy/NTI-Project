"""Adapter around the project's existing pricing engine."""

from src.pricing_engine import recommend_price


OBJECTIVES = {"maximize_revenue": "Maximize revenue", "maximize_profit": "Maximize profit", "balanced": "Balanced"}


def simulate(model_service, request):
    result, simulation = recommend_price(
        model_service.model,
        request.product_category,
        request.cost_price,
        request.current_price,
        request.competitor_price,
        request.inventory,
        int(request.promotion),
        request.season,
        request.customer_rating,
        OBJECTIVES[request.objective.value],
        request.date.isoformat(),
    )
    return result, simulation