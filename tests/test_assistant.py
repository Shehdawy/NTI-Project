from src.assistant import RecommendationContext, explain_recommendation


def test_explain_recommendation_mentions_price_and_objective():
    context = RecommendationContext(
        recommended_price=10.5,
        predicted_demand=42.3,
        expected_revenue=444.15,
        current_price=9.58,
        objective="Maximize profit",
        competitor_price=9.87,
        expected_profit=140.2,
        profit_margin=0.315,
        cost_price=6.23,
    )
    sentences = explain_recommendation(context)
    assert len(sentences) >= 3
    assert any("10.50" in sentence for sentence in sentences)
    assert any("profit" in sentence.lower() for sentence in sentences)


def test_explain_recommendation_handles_missing_profit():
    context = RecommendationContext(
        recommended_price=10.5,
        predicted_demand=42.3,
        expected_revenue=444.15,
        current_price=9.58,
        objective="Maximize revenue",
    )
    sentences = explain_recommendation(context)
    assert len(sentences) >= 2
    assert all(isinstance(sentence, str) and sentence for sentence in sentences)
