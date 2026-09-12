from datetime import datetime

KNOWN_QUESTIONS = [
    "Forecast Sales",
    "Recommend Price",
    "Compare Actions",
    "Why are sales expected to change?",
    "What should I do next?",
    "Explain Recommendation",
    "Business Insights",
    "Help",
]


def _parse_date(date_str, default=None):
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d")
    return default or datetime.today()


def answer_question(question: str, service, store: int = None, date: str = None,
                     objective: str = "balanced", base_price: float = 10.0,
                     elasticity: float = -1.2, unit_cost: float = None) -> dict:
    """
    `service` is a GenericStoreService instance for the logged-in store --
    always required, since every answer must be traceable to that store's
    own real data and trained model.
    """
    q = (question or "").strip().lower()
    dt = _parse_date(date)

    if store is None and getattr(service, "has_store_dimension", True):
        store_needed_keywords = ["forecast", "recommend", "compare", "why", "next", "explain"]
        if any(k in q for k in store_needed_keywords):
            return {
                "question": question,
                "answer": "I need a store number to answer that -- please include 'store' in your request.",
                "source": "assistant_validation",
            }

    if "help" in q:
        return {
            "question": question,
            "answer": "You can ask me: " + "; ".join(KNOWN_QUESTIONS),
            "source": "assistant_static",
        }

    if "business insight" in q or "insight" in q:
        insights = service.insights or {}
        summary_parts = []
        if "promotion" in insights:
            summary_parts.append(f"Promotions are associated with a {insights['promotion']['sales_lift_pct']}% sales lift.")
        if "seasonality" in insights:
            summary_parts.append(f"The strongest month historically is month {insights['seasonality']['best_month']}.")
        if "walmart_external" in insights:
            summary_parts.append(
                f"External benchmarking (Walmart, EXTERNAL DATA) shows a "
                f"{insights['walmart_external']['holiday_week_sales_lift_pct']}% holiday-week sales lift."
            )
        answer = " ".join(summary_parts) if summary_parts else "No business insights are available yet -- run notebook 06 first."
        return {"question": question, "answer": answer, "source": "business_insights.json"}

    if "forecast" in q and "recommend" not in q:
        forecast_val = service.forecast(store, dt)
        status = service.forecast_status(store, dt)
        where = f" for store {store}" if store is not None else ""
        return {
            "question": question,
            "answer": f"Forecasted sales{where} on {dt.date()}: {forecast_val:.2f} ({status}).",
            "source": "model_prediction",
        }

    if "why" in q:
        reasons = service.diagnose(store, dt)
        return {
            "question": question,
            "answer": "Possible contributing factors: " + " | ".join(reasons),
            "source": "diagnosis",
        }

    if "recommend price" in q or ("recommend" in q and "price" in q):
        forecast_val = service.forecast(store, dt)
        scenarios = service.simulate_pricing(base_price, elasticity, forecast_val, unit_cost)
        best = max(scenarios, key=lambda s: s["expected_revenue"])
        return {
            "question": question,
            "answer": f"Best-performing simulated price scenario: {best['scenario']} (price {best['price']}), "
                      f"expected revenue {best['expected_revenue']}. (pricing_method=SIMULATED, elasticity_source=ASSUMPTION)",
            "source": "pricing_simulation",
        }

    if "compare action" in q or ("compare" in q and "action" in q):
        forecast_val = service.forecast(store, dt)
        actions = service.compare_actions(forecast_val, base_price, elasticity, unit_cost)
        ranked = sorted(actions.items(), key=lambda kv: kv[1]["ranking"])
        lines = [f"{name}: revenue {data['revenue']} (rank {data['ranking']})" for name, data in ranked]
        return {"question": question, "answer": "; ".join(lines), "source": "decision_engine"}

    if "what should i do" in q or "next" in q:
        result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
        return {
            "question": question,
            "answer": f"Recommended Action: {result['recommended_action']}. Reason: {result['reason']}",
            "source": "decision_engine",
        }

    if "explain" in q:
        result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
        return {
            "question": question,
            "answer": result["reason"],
            "source": "decision_engine",
        }

    return {
        "question": question,
        "answer": "I didn't recognize that question. Try: " + "; ".join(KNOWN_QUESTIONS),
        "source": "assistant_fallback",
    }
