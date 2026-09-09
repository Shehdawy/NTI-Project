"""Streamlit dashboard for PricePilot AI.

Two ways to run this:
1. Local mode (default): loads the model and data directly from disk.
   This is what Streamlit Community Cloud uses.
2. API mode: set the PRICING_API_URL environment variable to point at a
   running FastAPI instance (see backend/main.py), and the dashboard will
   call that instead of loading the model itself.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

from src.assistant import RecommendationContext, explain_recommendation
from src.pricing_engine import recommend_price

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "prepared_sales_data.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"
API_BASE_URL = os.getenv("PRICING_API_URL")

OBJECTIVE_TO_API_VALUE = {
    "Maximize revenue": "maximize_revenue",
    "Maximize profit": "maximize_profit",
    "Balanced": "balanced",
}

PAGE_STYLE = """
<style>
    [data-testid="stSidebar"] {display: none;}
    .block-container {max-width: 1180px; padding-top: 3rem; padding-bottom: 4rem;}
    .eyebrow {
        color: #147d92; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .intro {color: #52616b; font-size: 1.05rem; margin: .35rem 0 2rem;}
    div[data-testid="stMetric"] {
        background: #f4f8f7; border: 1px solid #dbe9e6;
        padding: 1rem; border-radius: 8px;
    }
    div.stButton > button {border-radius: 6px; font-weight: 700; padding: .65rem 1.1rem;}
    .section-title {font-weight: 700; font-size: 1.05rem; margin-bottom: .25rem;}
    .section-caption {color: #6b7a80; font-size: .9rem; margin-bottom: 1rem;}
</style>
"""


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the prepared dataset used to seed sensible default input values."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_resource
def load_model():
    """Load the trained demand model (cached across reruns, not re-read every click)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metrics() -> pd.DataFrame | None:
    """Load the saved model comparison metrics, if available."""
    if not METRICS_PATH.exists():
        return None
    return pd.read_csv(METRICS_PATH)


def render_page_header() -> None:
    st.markdown('<div class="eyebrow">Decision support for smarter pricing</div>', unsafe_allow_html=True)
    st.title("PricePilot")
    st.markdown(
        '<p class="intro">A clear, data-informed price recommendation for every business decision.</p>',
        unsafe_allow_html=True,
    )


def render_scenario_form(data: pd.DataFrame) -> dict:
    """Render the input form and return the chosen scenario as a dict."""
    st.markdown('<div class="section-title">🧭 Set the scenario</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-caption">Enter the current business conditions. '
        "PricePilot will compare realistic price points and return one recommended option.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        left, right = st.columns(2, gap="large")
        with left:
            category = st.selectbox("Product category", sorted(data["product_category"].dropna().unique()))
            cost = st.number_input("Cost price", min_value=0.01, value=float(data["cost_price"].median()), step=1.0)
            current = st.number_input("Current selling price", min_value=0.01, value=float(data["selling_price"].median()), step=1.0)
            competitor = st.number_input("Competitor price", min_value=0.01, value=float(data["competitor_price"].median()), step=1.0)
        with right:
            inventory = st.number_input("Inventory units", min_value=1, value=int(data["inventory"].median()), step=1)
            promotion = st.checkbox("Promotion active")
            season = st.selectbox("Season", sorted(data["season"].dropna().unique()))
            rating = st.slider("Customer rating", 1.0, 5.0, float(data["customer_rating"].median()), 0.1)
            decision_date = st.date_input("Decision date", value=data["date"].max().date())
            objective = st.selectbox("Business priority", list(OBJECTIVE_TO_API_VALUE.keys()))

    return {
        "category": category,
        "cost": cost,
        "current": current,
        "competitor": competitor,
        "inventory": inventory,
        "promotion": promotion,
        "season": season,
        "rating": rating,
        "decision_date": decision_date,
        "objective": objective,
    }


def get_recommendation(scenario: dict) -> tuple[dict, pd.DataFrame]:
    """Fetch a recommendation, either from the API (if configured) or the local model."""
    if API_BASE_URL:
        payload = {
            "product_category": scenario["category"],
            "date": str(scenario["decision_date"]),
            "current_price": scenario["current"],
            "cost_price": scenario["cost"],
            "competitor_price": scenario["competitor"],
            "inventory": scenario["inventory"],
            "promotion": scenario["promotion"],
            "season": scenario["season"],
            "customer_rating": scenario["rating"],
            "objective": OBJECTIVE_TO_API_VALUE[scenario["objective"]],
        }
        response = requests.post(f"{API_BASE_URL.rstrip('/')}/api/v1/recommend-price", json=payload, timeout=15)
        response.raise_for_status()
        body = response.json()
        result = body["recommendation"]
        simulation = pd.DataFrame(body["scenarios"]).rename(columns={"price": "candidate_price"})
        return result, simulation

    result, simulation = recommend_price(
        load_model(),
        scenario["category"],
        scenario["cost"],
        scenario["current"],
        scenario["competitor"],
        scenario["inventory"],
        int(scenario["promotion"]),
        scenario["season"],
        scenario["rating"],
        scenario["objective"],
        str(scenario["decision_date"]),
    )
    return result, simulation


def render_result_metrics(result: dict) -> None:
    cards = st.columns(5)
    cards[0].metric("Recommended price", f"{result['recommended_price']:,.2f}")
    cards[1].metric("Predicted demand", f"{result['predicted_demand']:,.1f} units")
    cards[2].metric("Expected revenue", f"{result['expected_revenue']:,.2f}")
    cards[3].metric("Expected profit", f"{result['expected_profit']:,.2f}" if result["expected_profit"] is not None else "—")
    cards[4].metric("Profit margin", f"{result['profit_margin']:.1%}" if result["profit_margin"] is not None else "—")


def render_assistant_panel(scenario: dict, result: dict) -> None:
    """Render the small rule-based assistant that explains the recommendation.

    This is a template-driven explanation (see src/assistant.py) -- not a
    call to an external LLM -- so it works with no API key and no network
    access, and always explains the numbers actually shown above.
    """
    st.markdown('<div class="section-title">🤖 PricePilot Assistant</div>', unsafe_allow_html=True)
    context = RecommendationContext(
        recommended_price=result["recommended_price"],
        predicted_demand=result["predicted_demand"],
        expected_revenue=result["expected_revenue"],
        current_price=scenario["current"],
        objective=scenario["objective"],
        competitor_price=scenario["competitor"],
        expected_profit=result["expected_profit"],
        profit_margin=result["profit_margin"],
        cost_price=scenario["cost"],
    )
    explanation = " ".join(explain_recommendation(context))
    with st.chat_message("assistant"):
        st.write(explanation)


def render_scenario_chart(simulation: pd.DataFrame) -> None:
    chart_columns = ["predicted_demand", "expected_revenue"]
    if "expected_profit" in simulation and simulation["expected_profit"].notna().any():
        chart_columns.append("expected_profit")
    st.markdown('<div class="section-title">📈 Candidate prices compared</div>', unsafe_allow_html=True)
    st.line_chart(simulation.set_index("candidate_price")[chart_columns])
    with st.expander("View all tested candidate prices"):
        st.dataframe(simulation, use_container_width=True, hide_index=True)


def render_model_performance(metrics: pd.DataFrame | None) -> None:
    if metrics is None:
        return
    with st.expander("ℹ️ About the underlying model"):
        st.caption(
            "These are backtested metrics from a chronological train/test split, "
            "not a guarantee of future accuracy."
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="PricePilot | Decision Desk", page_icon="$", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    render_page_header()

    if not DATA_PATH.exists():
        st.error("The prepared dataset is missing. Run the training preparation step first.")
        st.stop()
    if not MODEL_PATH.exists():
        st.error("The trained model is missing. Expected models/demand_model.joblib.")
        st.stop()

    data = load_data()
    scenario = render_scenario_form(data)

    if st.button("Generate recommendation", type="primary", use_container_width=True):
        try:
            result, simulation = get_recommendation(scenario)
        except requests.RequestException:
            st.error(f"Pricing API is unavailable at {API_BASE_URL}. Remove PRICING_API_URL for local model mode.")
            return
        except (ValueError, KeyError) as error:
            st.error(f"Please check the inputs: {error}")
            return

        st.divider()
        render_result_metrics(result)
        render_assistant_panel(scenario, result)
        render_scenario_chart(simulation)

    st.divider()
    render_model_performance(load_model_metrics())


if __name__ == "__main__":
    main()
