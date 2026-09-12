import json
import os
from datetime import timedelta

import joblib
import pandas as pd


class GenericModelNotFoundError(Exception):
    pass


class GenericStoreService:

    def __init__(self, account_dir: str):
        self.account_dir = account_dir
        self.model = None
        self.feature_config = None
        self.insights = None
        self.history = None
        self._load()

    def _load(self):
        cfg_path = os.path.join(self.account_dir, "feature_config.json")
        model_path = os.path.join(self.account_dir, "model.pkl")
        if not (os.path.exists(cfg_path) and os.path.exists(model_path)):
            raise GenericModelNotFoundError(
                "No trained model found for this account yet -- upload your sales data first."
            )
        with open(cfg_path) as f:
            self.feature_config = json.load(f)
        self.model = joblib.load(model_path)

        insights_path = os.path.join(self.account_dir, "business_insights.json")
        if os.path.exists(insights_path):
            with open(insights_path) as f:
                self.insights = json.load(f)

        history_path = os.path.join(self.account_dir, "history.csv")
        if os.path.exists(history_path):
            self.history = pd.read_csv(history_path)

    @property
    def model_info(self):
        """Shaped like RetailService.model_info so the dashboard can show
        the same 'model loaded' summary regardless of which service is active."""
        return {
            "model_name": "Random Forest (trained on your data)",
            "features": self.feature_config["features"],
            "target": self.feature_config["target"],
            "metrics": {k: v for k, v in self.feature_config["metrics"].items() if v is not None},
        }

    @property
    def has_store_dimension(self):
        return bool(self.feature_config.get("has_store_dimension"))

    @property
    def valid_store_ids(self):
        if self.has_store_dimension and self.feature_config.get("store_map"):
            return set(self.feature_config["store_map"].keys())
        return set()

    def _build_row(self, store, date, promo=0, holiday=0):
        row = {
            "Year": date.year, "Month": date.month,
            "Week": date.isocalendar()[1], "Day": date.day,
            "DayOfWeek": date.isoweekday(),
            "IsWeekend": int(date.isoweekday() in (6, 7)),
        }
        if self.feature_config.get("has_promo"):
            row["Promo"] = int(promo)
        if self.feature_config.get("has_holiday"):
            row["Holiday"] = int(holiday)
        if self.has_store_dimension:
            store_map = self.feature_config["store_map"]
            if store is None or str(store) not in store_map:
                raise ValueError(f"Unknown store '{store}' for this account.")
            row["StoreCode"] = store_map[str(store)]
        features = self.feature_config["features"]
        return pd.DataFrame([[row[f] for f in features]], columns=features)

    def forecast(self, store, date, promo=0, holiday=0):
        X = self._build_row(store, date, promo, holiday)
        pred = float(self.model.predict(X)[0])
        return max(pred, 0.0)

    def forecast_status(self, store, date, promo=0):
        current = self.forecast(store, date, promo=promo)
        avg = None
        if self.insights and "overview" in self.insights:
            avg = self.insights["overview"].get("avg_sales_all")
        if not avg:
            return "Unknown"
        diff_pct = (current / avg - 1) * 100
        if diff_pct > 5:
            return "Increasing"
        if diff_pct < -5:
            return "Decreasing"
        return "Stable"

    def forecast_range(self, store, start_date, end_date, promo=0):
        results = []
        d = start_date
        while d <= end_date:
            forecast_val = self.forecast(store, d, promo=promo)
            results.append({
                "date": d.strftime("%Y-%m-%d"),
                "forecast_sales": round(forecast_val, 2),
                "forecast_status": self.forecast_status(store, d, promo=promo),
            })
            d += timedelta(days=1)
        return results

    def diagnose(self, store, date):
        reasons = []
        if date.isoweekday() in (6, 7):
            reasons.append("This date falls on a weekend, which affects sales in your historical data.")
        if self.insights and "seasonality" in self.insights:
            if self.insights["seasonality"].get("best_month") == date.month:
                reasons.append("This is historically your strongest month.")
        if self.feature_config.get("has_promo"):
            reasons.append("Promotions have a measurable effect in your data (see Business Insights).")
        if self.has_store_dimension and self.insights and "store_performance" in self.insights:
            top_stores = self.insights["store_performance"].get("top_5", {})
            if store in top_stores:
                reasons.append(f"'{store}' is historically one of your top-performing locations.")
        if not reasons:
            reasons.append("No strong deviation factors identified for this date.")
        return reasons

    def simulate_pricing(self, base_price, elasticity, baseline_demand, unit_cost=None):
        scenarios_pct = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        rows = []
        baseline_revenue = baseline_demand * base_price
        baseline_profit = (base_price - unit_cost) * baseline_demand if unit_cost is not None else None
        for pct in scenarios_pct:
            new_price = base_price * (1 + pct)
            simulated_demand = max(baseline_demand * (1 + elasticity * pct), 0)
            expected_revenue = simulated_demand * new_price
            revenue_change_pct = (expected_revenue / baseline_revenue - 1) * 100 if baseline_revenue else 0.0
            row = {
                "scenario": "Baseline" if pct == 0 else f"{pct*100:+.0f}%",
                "price": round(new_price, 2),
                "simulated_demand": round(simulated_demand, 2),
                "expected_revenue": round(expected_revenue, 2),
                "revenue_change_pct": round(revenue_change_pct, 2),
                "pricing_method": "SIMULATED",
                "elasticity_source": "ASSUMPTION",
            }
            if unit_cost is not None:
                expected_profit = (new_price - unit_cost) * simulated_demand
                profit_change_pct = (expected_profit / baseline_profit - 1) * 100 if baseline_profit else 0.0
                row["expected_profit"] = round(expected_profit, 2)
                row["profit_change_pct"] = round(profit_change_pct, 2)
            rows.append(row)
        return rows

    def compare_actions(self, baseline_demand, base_price, elasticity, unit_cost=None):
        PROMO_LIFT = 0.10  # ASSUMPTION

        def _impact(demand, price):
            revenue = demand * price
            profit = (price - unit_cost) * demand if unit_cost is not None else None
            return revenue, profit

        base_revenue, base_profit = _impact(baseline_demand, base_price)
        actions = {}

        actions["No Action"] = {"demand": round(baseline_demand, 2), "price": base_price, "revenue": round(base_revenue, 2)}

        promo_demand = baseline_demand * (1 + PROMO_LIFT)
        promo_revenue, promo_profit = _impact(promo_demand, base_price)
        actions["Promotion"] = {"demand": round(promo_demand, 2), "price": base_price, "revenue": round(promo_revenue, 2)}

        inc_price = base_price * 1.10
        inc_demand = max(baseline_demand * (1 + elasticity * 0.10), 0)
        inc_revenue, inc_profit = _impact(inc_demand, inc_price)
        actions["Price Increase"] = {"demand": round(inc_demand, 2), "price": round(inc_price, 2), "revenue": round(inc_revenue, 2)}

        dec_price = base_price * 0.90
        dec_demand = max(baseline_demand * (1 + elasticity * -0.10), 0)
        dec_revenue, dec_profit = _impact(dec_demand, dec_price)
        actions["Price Decrease"] = {"demand": round(dec_demand, 2), "price": round(dec_price, 2), "revenue": round(dec_revenue, 2)}

        combo_demand = max(baseline_demand * (1 + PROMO_LIFT) * (1 + elasticity * -0.05), 0)
        combo_price = base_price * 0.95
        combo_revenue, combo_profit = _impact(combo_demand, combo_price)
        actions["Promotion + Price Change"] = {"demand": round(combo_demand, 2), "price": round(combo_price, 2), "revenue": round(combo_revenue, 2)}

        profits = {"No Action": base_profit, "Promotion": promo_profit, "Price Increase": inc_profit,
                   "Price Decrease": dec_profit, "Promotion + Price Change": combo_profit}

        for name, data in actions.items():
            data["revenue_change_pct"] = round((data["revenue"] / base_revenue - 1) * 100, 2) if base_revenue else 0.0
            if unit_cost is not None:
                data["profit"] = round(profits[name], 2)
                data["profit_change_pct"] = round((profits[name] / base_profit - 1) * 100, 2) if base_profit else 0.0

        ranked = sorted(actions.items(), key=lambda kv: kv[1]["revenue"], reverse=True)
        for rank, (name, _) in enumerate(ranked, start=1):
            actions[name]["ranking"] = rank

        return actions

    def recommend(self, store, date, objective, base_price, elasticity, unit_cost=None):
        baseline_demand = self.forecast(store, date)
        status = self.forecast_status(store, date)
        reasons = self.diagnose(store, date)
        actions = self.compare_actions(baseline_demand, base_price, elasticity, unit_cost)

        key = "demand" if objective == "maximize_demand" else "revenue"
        best_action = max(actions.items(), key=lambda kv: kv[1][key])
        recommended_action, impact = best_action

        reason = (
            f"'{recommended_action}' produces the highest {key} ({impact[key]:.2f}) "
            f"among the compared actions for objective '{objective}', given a baseline "
            f"forecast of {baseline_demand:.2f} units ({status})."
        )

        return {
            "store": store,
            "date": date.strftime("%Y-%m-%d"),
            "forecast_sales": round(baseline_demand, 2),
            "forecast_status": status,
            "diagnosis": reasons,
            "available_actions": actions,
            "recommended_action": recommended_action,
            "expected_impact": impact,
            "reason": reason,
        }
