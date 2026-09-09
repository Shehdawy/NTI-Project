"""A small, rule-based "assistant" that explains a price recommendation.

This is deliberately NOT a call to an external LLM. It is a template-driven
natural-language generator: it looks at the numbers PricePilot already
computed (recommended price vs. current price, margin, demand, objective)
and turns them into a short, plain-language explanation a non-technical
stakeholder can read. Same inputs always produce the same explanation, there
is no network call, and no API key is required.
"""

from dataclasses import dataclass


@dataclass
class RecommendationContext:
    """Everything the assistant needs to explain one recommendation.

    This mirrors the fields already produced by
    :func:`src.pricing_engine.recommend_price`, plus the handful of request
    inputs (current price, cost, competitor price, objective) needed to
    compare the recommendation against the scenario the user described.
    """

    recommended_price: float
    predicted_demand: float
    expected_revenue: float
    current_price: float
    objective: str
    competitor_price: float | None = None
    expected_profit: float | None = None
    profit_margin: float | None = None
    cost_price: float | None = None


def _describe_price_move(context: RecommendationContext) -> str:
    """One sentence comparing the recommended price to the current price."""
    delta = context.recommended_price - context.current_price
    percent = abs(delta) / context.current_price * 100 if context.current_price else 0

    if abs(delta) < 0.01:
        return "It keeps the price right where it is today."
    direction = "raising" if delta > 0 else "lowering"
    return (
        f"It suggests {direction} the price from {context.current_price:,.2f} "
        f"to {context.recommended_price:,.2f} ({percent:.1f}% {'higher' if delta > 0 else 'lower'})."
    )


def _describe_competitor_position(context: RecommendationContext) -> str | None:
    """One sentence on how the recommended price sits versus the competitor."""
    if context.competitor_price is None or context.competitor_price <= 0:
        return None
    gap_percent = (context.recommended_price - context.competitor_price) / context.competitor_price * 100
    if abs(gap_percent) < 2:
        return "That price is essentially in line with the competitor's price."
    position = "above" if gap_percent > 0 else "below"
    return f"That puts you {abs(gap_percent):.1f}% {position} the competitor's price."


def _describe_margin(context: RecommendationContext) -> str | None:
    """One sentence on how healthy the resulting profit margin looks."""
    if context.profit_margin is None:
        return None
    margin_percent = context.profit_margin * 100
    if margin_percent < 10:
        health = "thin — worth double-checking the cost assumption"
    elif margin_percent < 30:
        health = "reasonable"
    else:
        health = "healthy"
    return f"At this price, the expected profit margin is about {margin_percent:.0f}%, which looks {health}."


def _describe_objective(context: RecommendationContext) -> str:
    """One sentence naming which business goal drove the choice."""
    objective_text = {
        "Maximize revenue": "chosen to bring in the most total sales revenue",
        "Maximize profit": "chosen to bring in the most total profit",
        "Balanced": "chosen as a balance between revenue and profit, rather than maximizing either alone",
    }
    reason = objective_text.get(context.objective, f"chosen to satisfy the '{context.objective}' objective")
    return f"Among the price points tested, this one was {reason}."


def explain_recommendation(context: RecommendationContext) -> list[str]:
    """Build a short list of plain-language sentences explaining a recommendation.

    Each sentence stands on its own (safe to render as separate bullet
    points or joined into a paragraph). The output is fully determined by
    the numbers in ``context`` — nothing is invented or looked up.
    """
    sentences = [
        _describe_price_move(context),
        _describe_objective(context),
    ]
    for sentence in (_describe_competitor_position(context), _describe_margin(context)):
        if sentence:
            sentences.append(sentence)
    sentences.append(
        f"Expected demand at this price is about {context.predicted_demand:,.1f} units, "
        f"for expected revenue of {context.expected_revenue:,.2f}."
    )
    return sentences
