"""
AI Retail Decision Support — Streamlit App

A single, polished flow: a store owner creates an account, uploads their own
sales data (any schema), gets a model trained on it in seconds, and uses
real forecasting / pricing / decision tools on their own numbers.

No business logic lives here -- this file only renders inputs/outputs.
Every number shown comes from api/generic_training.py + api/generic_service.py
+ api/assistant.py.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Robust import of the shared business-logic package regardless of the
# working directory the app is launched from.
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.generic_service import GenericStoreService, GenericModelNotFoundError  # noqa: E402
from api.generic_training import train_store_model, InsufficientDataError  # noqa: E402
from api.assistant import answer_question, KNOWN_QUESTIONS  # noqa: E402
from api import accounts  # noqa: E402

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# Cohesive color palette used across all charts and CSS accents
COLOR_PRIMARY = "#6C5CE7"
COLOR_PRIMARY_DARK = "#4834D4"
COLOR_ACCENT = "#00B894"
COLOR_WARN = "#E17055"
COLOR_BG_CARD = "#FFFFFF"
CHART_SEQUENCE = ["#6C5CE7", "#00B894", "#0984E3", "#FDCB6E", "#E17055", "#00CEC9"]


# --------------------------------------------------------------------------
# Page setup & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail AI — Forecast & Decide",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Poppins', sans-serif; }}

    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}}

    .hero {{
        padding: 2.6rem 3rem;
        border-radius: 22px;
        background: linear-gradient(135deg, {COLOR_PRIMARY_DARK} 0%, {COLOR_PRIMARY} 55%, #00cec9 130%);
        color: #ffffff;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        box-shadow: 0 12px 30px rgba(76, 52, 212, 0.25);
    }}
    .hero-text h1 {{margin: 0 0 0.4rem 0; font-size: 2.3rem; font-weight: 800;}}
    .hero-text p {{margin: 0; opacity: 0.92; font-size: 1.08rem; max-width: 520px;}}
    .hero-badges {{margin-top: 1rem;}}
    .hero-badge {{
        display: inline-block; background: rgba(255,255,255,0.18);
        padding: 5px 14px; border-radius: 999px; font-size: 0.82rem;
        margin-right: 8px; font-weight: 600;
    }}

    .feature-card {{
        background: {COLOR_BG_CARD};
        border-radius: 18px;
        padding: 1.6rem 1.4rem;
        border: 1px solid #ECECF9;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.05);
        height: 100%;
        transition: transform 0.15s ease;
    }}
    .feature-card .icon {{font-size: 2.1rem; margin-bottom: 0.5rem;}}
    .feature-card h4 {{margin: 0 0 0.35rem 0; font-family: 'Poppins', sans-serif;}}
    .feature-card p {{margin: 0; color: #555; font-size: 0.92rem; line-height: 1.4;}}

    .recommend-box {{
        background: linear-gradient(135deg, #0f5132 0%, {COLOR_ACCENT} 110%);
        color: #ffffff;
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(0, 184, 148, 0.25);
    }}
    .recommend-box h2 {{margin-top: 0; margin-bottom: 0.4rem; font-family: 'Poppins', sans-serif;}}
    .recommend-box p {{opacity: 0.95; font-size: 1.04rem;}}

    .tag-primary   {{background:#EFEBFF; color:{COLOR_PRIMARY_DARK}; padding:3px 12px; border-radius:999px; font-size:0.78rem; font-weight:700;}}
    .tag-assumption{{background:#FDECEA; color:#B02A37; padding:3px 12px; border-radius:999px; font-size:0.78rem; font-weight:700;}}
    .tag-simulated {{background:#E7F6F2; color:#0f5132; padding:3px 12px; border-radius:999px; font-size:0.78rem; font-weight:700;}}

    section[data-testid="stSidebar"] {{border-right: 1px solid #ECECF9; background: #FBFAFF;}}

    .stTabs [data-baseweb="tab-list"] {{gap: 4px;}}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }}

    div[data-testid="stMetric"] {{
        background: {COLOR_BG_CARD};
        border: 1px solid #ECECF9;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 12px rgba(31,41,55,0.04);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLOR_PRIMARY_DARK}, {COLOR_PRIMARY});
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STORE_HERO_SVG = """
<svg width="150" height="150" viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="75" cy="75" r="72" fill="rgba(255,255,255,0.10)"/>
  <circle cx="75" cy="75" r="55" fill="rgba(255,255,255,0.12)"/>
  <rect x="42" y="55" width="66" height="46" rx="6" fill="#ffffff" opacity="0.95"/>
  <path d="M42 55 L50 34 H100 L108 55 Z" fill="#FDCB6E"/>
  <rect x="50" y="70" width="14" height="31" rx="2" fill="#6C5CE7"/>
  <rect x="68" y="60" width="14" height="41" rx="2" fill="#00B894"/>
  <rect x="86" y="66" width="14" height="35" rx="2" fill="#0984E3"/>
  <path d="M46 48 L60 30 M60 30 L74 46 M90 46 L104 30 M104 30 L118 48" stroke="none"/>
  <circle cx="112" cy="42" r="14" fill="#E17055"/>
  <path d="M105 42 L110 47 L120 36" stroke="#ffffff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def render_hero(subtitle_extra=""):
    logo_path = os.path.join(PROJECT_ROOT, "assets", "logo.png")
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        graphic_html = f'<img src="data:image/png;base64,{b64}" style="width:120px;height:120px;object-fit:contain;" />'
    else:
        graphic_html = STORE_HERO_SVG

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-text">
                <h1>🛍️ Your AI Retail Co-Pilot</h1>
                <p>Upload your own sales data and get real forecasts, price simulations,
                and clear "what to do next" recommendations in minutes.{subtitle_extra}</p>
                <div class="hero-badges">
                    <span class="hero-badge">📤 Bring Your Own Data</span>
                    <span class="hero-badge">🧠 Trained Just For You</span>
                    <span class="hero-badge">🔒 Private To Your Account</span>
                </div>
            </div>
            <div>{graphic_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards():
    cols = st.columns(3)
    features = [
        ("📤", "Upload Your Sales History", "Any CSV export works — we'll help you map the columns that matter."),
        ("🧠", "AI Learns Your Patterns", "A real machine learning model trains on your own data in seconds."),
        ("🎯", "Get a Clear Recommendation", "Forecasts, price scenarios, and a plain-English 'do this next'."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f"""<div class="feature-card"><div class="icon">{icon}</div>
            <h4>{title}</h4><p>{desc}</p></div>""",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def render_auth_form():
    render_hero()
    st.write("")
    render_feature_cards()
    st.write("")
    st.markdown("### Get started — log in or create your store's account")
    tab_login, tab_register = st.tabs(["Log In", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary")
        if submitted:
            if accounts.verify(u, p):
                st.session_state["logged_in_user"] = u
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab_register:
        with st.form("register_form"):
            store_name = st.text_input("Store / Business Name")
            u2 = st.text_input("Choose a Username")
            p2 = st.text_input("Choose a Password", type="password")
            submitted2 = st.form_submit_button("Create Account", type="primary")
        if submitted2:
            ok, msg = accounts.register(u2, p2, store_name)
            if ok:
                st.success(msg + " You can now log in from the 'Log In' tab.")
            else:
                st.error(msg)


# --------------------------------------------------------------------------
# Upload & train
# --------------------------------------------------------------------------
def render_upload_and_train(acc_dir, retrain=False):
    key_suffix = "retrain" if retrain else "initial"

    if not retrain:
        render_hero(" You're almost there — just upload your data below.")
        with st.container(border=True):
            st.markdown("#### Step 1 — Upload your CSV")
            st.caption("At minimum: a date column and a sales/revenue column. Promotions, holidays, and multiple locations are optional.")
            uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key=f"uploader_{key_suffix}")
    else:
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key=f"uploader_{key_suffix}")

    if uploaded is None:
        return

    try:
        df_preview = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file as a CSV: {exc}")
        return

    if df_preview.empty:
        st.error("That file has no rows.")
        return

    with st.container(border=True):
        st.markdown("#### Step 2 — Confirm your columns")
        st.dataframe(df_preview.head(8), use_container_width=True)

        columns = list(df_preview.columns)
        col1, col2 = st.columns(2)
        date_col = col1.selectbox("Which column is the DATE?", columns, key=f"date_col_{key_suffix}")
        sales_col = col2.selectbox("Which column is SALES / REVENUE?", columns, key=f"sales_col_{key_suffix}")

        none_option = "(none)"
        with st.expander("Optional: map more columns (improves accuracy)"):
            promo_col = st.selectbox("Promotion flag column", [none_option] + columns, key=f"promo_col_{key_suffix}")
            holiday_col = st.selectbox("Holiday flag column", [none_option] + columns, key=f"holiday_col_{key_suffix}")
            store_col = st.selectbox(
                "Store / branch name column (only if you have multiple locations)",
                [none_option] + columns, key=f"store_col_{key_suffix}",
            )

    with st.container(border=True):
        st.markdown("#### Step 3 — Train your model")
        if st.button("🚀 Train My Model", type="primary", key=f"train_btn_{key_suffix}", use_container_width=True):
            with st.spinner("Training your model on your data..."):
                try:
                    cfg = train_store_model(
                        df_preview, date_col=date_col, sales_col=sales_col,
                        promo_col=None if promo_col == none_option else promo_col,
                        holiday_col=None if holiday_col == none_option else holiday_col,
                        store_col=None if store_col == none_option else store_col,
                        output_dir=acc_dir,
                    )
                except (InsufficientDataError, ValueError) as exc:
                    st.error(str(exc))
                    return

            metrics = cfg["metrics"]
            metrics_str = f"MAE {metrics['MAE']:.2f}, RMSE {metrics['RMSE']:.2f}"
            if metrics.get("R2") is not None:
                metrics_str += f", R² {metrics['R2']:.3f}"
            st.balloons()
            st.success(f"🎉 Model trained on {cfg['rows_trained']} rows! {metrics_str}")
            st.cache_resource.clear()
            st.rerun()


# --------------------------------------------------------------------------
# Session / login gate
# --------------------------------------------------------------------------
if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if not st.session_state["logged_in_user"]:
    render_auth_form()
    st.stop()

username = st.session_state["logged_in_user"]
store_display_name = accounts.get_store_name(username)


@st.cache_resource(show_spinner="Loading your store's model...")
def load_generic_service(acc_dir):
    return GenericStoreService(acc_dir)


acc_dir = accounts.account_dir(username)
try:
    service = load_generic_service(acc_dir)
except GenericModelNotFoundError:
    col_a, col_b = st.columns([5, 1])
    col_a.markdown(f"### Welcome, {store_display_name} 👋")
    if col_b.button("Log out"):
        st.session_state["logged_in_user"] = None
        st.rerun()
    render_upload_and_train(acc_dir)
    st.stop()


# --------------------------------------------------------------------------
# Header (logged in, model ready)
# --------------------------------------------------------------------------
render_hero(f" Welcome back, {store_display_name}.")

col_a, col_b = st.columns([5, 1])
m_info = service.model_info
metrics_bits = []
for k, v in m_info["metrics"].items():
    metrics_bits.append(f"{k} {v:.2f}" if isinstance(v, (int, float)) else f"{k} {v}")
col_a.caption(
    f"Model trained on **{service.feature_config['rows_trained']} rows** "
    f"of your data · {' · '.join(metrics_bits)}"
)
if col_b.button("Log out", use_container_width=True):
    st.session_state["logged_in_user"] = None
    st.rerun()

with st.expander("🔄 Upload new data / retrain your model"):
    render_upload_and_train(acc_dir, retrain=True)


# --------------------------------------------------------------------------
# Sidebar — scenario inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Scenario Inputs")

    if getattr(service, "has_store_dimension", False):
        store_options = sorted(service.valid_store_ids)
        store = st.selectbox("Store / Location", store_options)
    else:
        store = None
        st.caption("Single-location account — no store selector needed.")

    forecast_date = st.date_input("Forecast Date", value=date.today())

    objective_labels = {
        "balanced": "Balanced",
        "maximize_revenue": "Maximize Revenue",
        "maximize_demand": "Maximize Demand",
    }
    objective = st.selectbox(
        "Business Objective", options=list(objective_labels.keys()),
        format_func=lambda k: objective_labels[k],
    )

    st.markdown("---")
    st.subheader("💵 Pricing Assumptions")
    st.caption("No real price exists in the data — these are disclosed, adjustable assumptions.")

    base_price = st.number_input("Base Price ($)", min_value=0.01, value=10.0, step=0.5)
    elasticity = st.slider(
        "Price Elasticity of Demand", min_value=-3.0, max_value=-0.1, value=-1.2, step=0.1,
        help="How much demand changes for a 1% price change. More negative = more price-sensitive.",
    )
    include_cost = st.checkbox("Include profit calculations", value=True)
    unit_cost = None
    if include_cost:
        unit_cost = st.number_input("Unit Cost ($)", min_value=0.0, value=6.0, step=0.5)

    st.markdown("---")
    run_clicked = st.button("🔍 Run Analysis", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Run analysis
# --------------------------------------------------------------------------
if run_clicked:
    try:
        dt = datetime(forecast_date.year, forecast_date.month, forecast_date.day)
        result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
        scenarios = service.simulate_pricing(base_price, elasticity, result["forecast_sales"], unit_cost)
        st.session_state["result"] = result
        st.session_state["scenarios"] = scenarios
        st.session_state["last_inputs"] = dict(
            store=store, date=forecast_date, objective=objective,
            base_price=base_price, elasticity=elasticity, unit_cost=unit_cost,
        )
    except ValueError as exc:
        st.session_state["result"] = None
        st.error(str(exc))

result = st.session_state.get("result")
scenarios = st.session_state.get("scenarios")

tab_forecast, tab_pricing, tab_actions, tab_recommend, tab_insights, tab_assistant = st.tabs(
    ["📊 Forecast & Diagnosis", "💰 Price Simulation", "⚖️ Compare Actions",
     "🎯 Recommendation", "📈 Business Insights", "🤖 Ask the Assistant"]
)


# --------------------------------------------------------------------------
# Tab 1 — Forecast & Diagnosis
# --------------------------------------------------------------------------
with tab_forecast:
    if not result:
        st.info("👈 Set your scenario in the sidebar and click **Run Analysis**.")
    else:
        stored_store = st.session_state["last_inputs"]["store"]
        cols = st.columns(3 if stored_store is not None else 2)
        cols[0].metric("Forecasted Sales", f"${result['forecast_sales']:,.2f}")
        status_icon = {"Increasing": "📈", "Decreasing": "📉", "Stable": "➖", "Unknown": "❓"}
        cols[1].metric("Trend Status", f"{status_icon.get(result['forecast_status'],'')} {result['forecast_status']}")
        if stored_store is not None:
            cols[2].metric("Store", stored_store)

        if service.history is not None and len(service.history) > 5:
            st.markdown("#### Recent Sales History")
            hist = service.history.copy()
            date_col_name = service.feature_config["date_col"]
            target_col_name = service.feature_config["target"]
            hist[date_col_name] = pd.to_datetime(hist[date_col_name])
            if stored_store is not None and "StoreName" in hist.columns:
                hist = hist[hist["StoreName"] == stored_store]
            hist = hist.sort_values(date_col_name).tail(60)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist[date_col_name], y=hist[target_col_name],
                mode="lines", name="Actual", line=dict(color=COLOR_PRIMARY, width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(forecast_date)], y=[result["forecast_sales"]],
                mode="markers", name="Forecast", marker=dict(color=COLOR_ACCENT, size=13, symbol="star"),
            ))
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Why this forecast?")
        for reason in result["diagnosis"]:
            st.markdown(f"- {reason}")

        st.caption("Forecast source: real, trained model prediction (not simulated).")


# --------------------------------------------------------------------------
# Tab 2 — Price Simulation
# --------------------------------------------------------------------------
with tab_pricing:
    if not scenarios:
        st.info("👈 Run an analysis first to see price scenarios.")
    else:
        st.markdown(
            '<span class="tag-simulated">SIMULATED</span> &nbsp; '
            '<span class="tag-assumption">ELASTICITY: ASSUMPTION</span>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(scenarios)

        fig = go.Figure()
        colors = [COLOR_ACCENT if s == "Baseline" else COLOR_PRIMARY for s in df["scenario"]]
        fig.add_trace(go.Bar(
            x=df["scenario"], y=df["expected_revenue"], marker_color=colors, name="Expected Revenue",
        ))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Expected Revenue ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        display_cols = ["scenario", "price", "simulated_demand", "expected_revenue", "revenue_change_pct"]
        if "expected_profit" in df.columns:
            display_cols += ["expected_profit", "profit_change_pct"]
        st.dataframe(
            df[display_cols].rename(columns={
                "scenario": "Scenario", "price": "Price ($)", "simulated_demand": "Demand",
                "expected_revenue": "Revenue ($)", "revenue_change_pct": "Revenue Δ%",
                "expected_profit": "Profit ($)", "profit_change_pct": "Profit Δ%",
            }),
            use_container_width=True, hide_index=True,
        )


# --------------------------------------------------------------------------
# Tab 3 — Compare Actions
# --------------------------------------------------------------------------
with tab_actions:
    if not result:
        st.info("👈 Run an analysis first to compare business actions.")
    else:
        actions_df = pd.DataFrame(result["available_actions"]).T.sort_values("ranking")
        actions_df.index.name = "Action"

        fig = go.Figure()
        colors = [COLOR_ACCENT if r == 1 else COLOR_PRIMARY for r in actions_df["ranking"]]
        fig.add_trace(go.Bar(
            y=actions_df.index, x=actions_df["revenue"], orientation="h", marker_color=colors,
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Revenue ($)", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(actions_df, use_container_width=True)


# --------------------------------------------------------------------------
# Tab 4 — Recommendation
# --------------------------------------------------------------------------
with tab_recommend:
    if not result:
        st.info("👈 Run an analysis first to get a recommendation.")
    else:
        st.markdown(
            f"""
            <div class="recommend-box">
                <h2>✅ Recommended Action: {result['recommended_action']}</h2>
                <p>{result['reason']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        impact = {k: v for k, v in result["expected_impact"].items() if k != "ranking"}
        cols = st.columns(len(impact))
        for col, (key, val) in zip(cols, impact.items()):
            label = key.replace("_", " ").title()
            display_val = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
            col.metric(label, display_val)


# --------------------------------------------------------------------------
# Tab 5 — Business Insights
# --------------------------------------------------------------------------
with tab_insights:
    insights = service.insights or {}
    st.markdown('<span class="tag-primary">PRIMARY DATA — Your Store</span>', unsafe_allow_html=True)

    if not insights:
        st.info("No insights yet — upload data and train a model first.")

    if "overview" in insights:
        ov = insights["overview"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Analyzed", f"{ov['rows_used']:,}")
        c2.metric("Date Range", f"{ov['date_range'][0]} → {ov['date_range'][1]}")
        c3.metric("Average Sales", f"${ov['avg_sales_all']:,.2f}")

    if "promotion" in insights:
        st.markdown("#### 🏷️ Promotion Impact")
        p = insights["promotion"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Sales — No Promo", f"${p['avg_no_promo']:,.2f}")
        c2.metric("Avg Sales — With Promo", f"${p['avg_with_promo']:,.2f}")
        c3.metric("Sales Lift", f"{p['sales_lift_pct']:+.1f}%")

    if "seasonality" in insights:
        st.markdown("#### 📅 Seasonality")
        s = insights["seasonality"]
        col1, col2 = st.columns(2)
        with col1:
            dow = {DAY_NAMES[int(k)]: v for k, v in s["avg_by_day_of_week"].items()}
            fig = px.bar(x=list(dow.keys()), y=list(dow.values()), color_discrete_sequence=[COLOR_PRIMARY])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Avg Sales by Day of Week",
                               plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            month = {MONTH_NAMES[int(k)]: v for k, v in s["avg_by_month"].items()}
            fig2 = px.bar(x=list(month.keys()), y=list(month.values()), color_discrete_sequence=[COLOR_ACCENT])
            fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Avg Sales by Month",
                                plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

    if "store_performance" in insights:
        st.markdown("#### 🏬 Store Performance")
        sp = insights["store_performance"]
        combined = {**sp["top_5"], **sp["bottom_5"]}
        fig3 = px.bar(
            x=list(combined.values()), y=list(combined.keys()), orientation="h",
            color=list(combined.values()), color_continuous_scale=["#E17055", "#00B894"],
        )
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="Average Sales ($)", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------------------------------
# Tab 6 — AI Assistant
# --------------------------------------------------------------------------
with tab_assistant:
    st.caption(
        "Deterministic assistant — every answer is computed from your real model and data "
        "above, no LLM required. Try: " + "; ".join(KNOWN_QUESTIONS[:4]) + "..."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("Ask about forecasts, pricing, or what to do next...")
    if question:
        st.session_state["chat_history"].append(("user", question))
        inputs = st.session_state.get("last_inputs", {})
        response = answer_question(
            question, service,
            store=inputs.get("store", store),
            date=str(inputs.get("date", forecast_date)),
            objective=inputs.get("objective", objective),
            base_price=inputs.get("base_price", base_price),
            elasticity=inputs.get("elasticity", elasticity),
            unit_cost=inputs.get("unit_cost", unit_cost),
        )
        st.session_state["chat_history"].append(("assistant", response["answer"]))
        st.rerun()
