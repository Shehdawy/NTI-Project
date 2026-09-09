"""Adapter around the project's existing pricing engine."""

from src.assistant import RecommendationContext, explain_recommendation
from src.pricing_engine import recommend_price

# Map the API's snake_case enum values to the human-readable objective
# names recommend_price() expects.
OBJECTIVE_LABELS = {
    "maximize_revenue": "Maximize revenue",
    "maximize_profit": "Maximize profit",
    "balanced": "Balanced",
}


def simulate(model_service, request):
    """Run the pricing simulation for one API request and return (result, simulation)."""
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
        OBJECTIVE_LABELS[request.objective.value],
        request.date.isoformat(),
    )
    return result, simulation


def explain(request, result: dict) -> list[str]:
    """Turn a recommendation result into a short plain-language explanation."""
    context = RecommendationContext(
        recommended_price=result["recommended_price"],
        predicted_demand=result["predicted_demand"],
        expected_revenue=result["expected_revenue"],
        current_price=request.current_price,
        objective=OBJECTIVE_LABELS[request.objective.value],
        competitor_price=request.competitor_price,
        expected_profit=result["expected_profit"],
        profit_margin=result["profit_margin"],
        cost_price=request.cost_price,
    )
    return explain_recommendation(context)
