"""Streamlit dashboard for PricePilot AI."""

from pathlib import Path
import os

import joblib
import pandas as pd
import requests
import streamlit as st

from src.pricing_engine import recommend_price

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "sales_data.csv"
MODEL_PATH = ROOT / "models" / "demand_model.pkl"
API_BASE_URL = os.getenv("PRICING_API_URL")


@st.cache_data
def load_metrics():
    return pd.read_csv(ROOT / "models" / "model_metrics.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="PricePilot | Decision Desk", page_icon="$", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        .block-container {max-width: 1180px; padding-top: 3rem; padding-bottom: 4rem;}
        .eyebrow {color: #147d92; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
        .intro {color: #52616b; font-size: 1.05rem; margin: .35rem 0 2rem;}
        div[data-testid="stMetric"] {background: #f4f8f7; border: 1px solid #dbe9e6; padding: 1rem; border-radius: 8px;}
        div.stButton > button {border-radius: 6px; font-weight: 700; padding: .65rem 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Decision support for smarter pricing</div>', unsafe_allow_html=True)
st.title("PricePilot")
st.markdown('<p class="intro">A clear, data-informed price recommendation for every business decision.</p>', unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("The supplied dataset is missing. Expected data/processed/sales_data.csv.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Expected models/demand_model.pkl.")
    st.stop()

data = load_data()
st.header("Set the scenario")
st.caption("Enter the current business conditions. PricePilot will compare realistic price points and return one recommended option.")
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
    objective = st.selectbox("Business priority", ["Maximize revenue", "Maximize profit", "Balanced"])

if st.button("Generate recommendation", type="primary", use_container_width=True):
        try:
            payload = {"product_category": category, "date": str(decision_date), "current_price": current, "cost_price": cost, "competitor_price": competitor, "inventory": inventory, "promotion": promotion, "season": season, "customer_rating": rating, "objective": {"Maximize revenue": "maximize_revenue", "Maximize profit": "maximize_profit", "Balanced": "balanced"}[objective]}
            if API_BASE_URL:
                response = requests.post(f"{API_BASE_URL.rstrip('/')}/recommend-price", json=payload, timeout=15)
                response.raise_for_status()
                body = response.json()
                result = body["recommendation"]
                simulation = pd.DataFrame(body["scenarios"]).rename(columns={"price": "candidate_price"})
            else:
                result, simulation = recommend_price(load_model(), category, cost, current, competitor, inventory, int(promotion), season, rating, objective, str(decision_date))
                simulation = simulation.rename(columns={"candidate_price": "candidate_price"})
            cards = st.columns(5)
            cards[0].metric("Recommended price", f"{result['recommended_price']:,.2f}")
            cards[1].metric("Predicted demand", f"{result['predicted_demand']:,.1f} units")
            cards[2].metric("Expected revenue", f"{result['expected_revenue']:,.2f}")
            cards[3].metric("Expected profit", f"{result['expected_profit']:,.2f}")
            cards[4].metric("Profit margin", f"{result['profit_margin']:.1%}")
            chart_columns = ["predicted_demand", "expected_revenue"]
            if "expected_profit" in simulation and simulation["expected_profit"].notna().any():
                chart_columns.append("expected_profit")
            st.line_chart(simulation.set_index("candidate_price")[chart_columns])
        except requests.RequestException:
            st.error(f"Pricing API is unavailable at {API_BASE_URL}. Remove PRICING_API_URL for single-link local model mode.")
        except (ValueError, KeyError) as error:
            st.error(f"Please check the inputs: {error}")