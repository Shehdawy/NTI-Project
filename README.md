# PricePilot AI — Retail Price Recommendation System

PricePilot AI predicts product demand and simulates prices to recommend a reasonable selling price. The model predicts expected units sold; business logic then compares candidate prices by expected revenue and profit. It does not claim to discover a mathematically perfect or autonomous price.

## Business Objective

Help a small business balance demand and margin using historical sales, product cost, competitor price, inventory, promotion, category, season, and customer rating.

## Dataset

The supplied dataset is `data/processed/sales_data.csv`, with 5,000 rows and 15 columns. It contains `product_id`, `product_category`, `date`, `cost_price`, `selling_price`, `competitor_price`, `units_sold`, `inventory`, `promotion`, `season`, `day_of_week`, `customer_rating`, `demand`, `revenue`, and `profit`. The project does not invent these fields. A machine-readable inspection is saved in `models/data_report.json` after training.

The data period is 2023-01-01 through 2024-12-31. Because the data ends before the current date, this application demonstrates the historical-data methodology and does not claim to predict current market prices.

## Project Architecture

```text
data/processed/sales_data.csv -> preprocessing -> demand models -> models/demand_model.pkl
                                                               |
                                                               v
                                                   Streamlit price simulation
```

Important files: `src/preprocessing.py`, `src/pricing_engine.py`, `src/dl_model.py`, `train_models.py`, and `app.py`.

## Machine Learning

The target is `units_sold`, so this is a regression problem. A mean-demand baseline, Linear Regression, Random Forest, and Gradient Boosting are compared using MAE, RMSE, and R². The final 20% of records by date is held out as the test set to reduce time leakage. Categorical fields use one-hot encoding and numeric fields are standardized in a scikit-learn pipeline. Revenue, profit, and `demand` are not model inputs because they are outcomes or duplicate the target.

## Deep Learning

`src/dl_model.py` contains a small Keras network with Dense, Dropout, and EarlyStopping. Install `requirements-dl.txt` to run it. TensorFlow is intentionally optional because classical models are usually a strong and easier-to-explain baseline for tabular business data; the deployed app does not depend on it.

## Pricing Recommendation Logic

The engine tests 25 prices between a current-price range and a competitor-aware range. For each candidate it predicts demand, caps demand at user-entered inventory, and calculates revenue, profit, and margin. It enforces a minimum price of cost plus 8% and avoids prices more than 20% above the competitor price. The user can maximize revenue, maximize profit, or choose a simple balanced objective. These are expected values, not guarantees.

## How To Run

```powershell
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The training command reads the supplied CSV, creates `models/demand_model.pkl`, model metrics, data report, test predictions, feature importance, EDA output, and the optional deep-learning result. Run it again whenever the data or training code changes.

## FastAPI Backend

The project also includes a lightweight REST API under `backend/`. Install its pinned serving dependencies with `python -m pip install -r backend/requirements.txt`, then run `python -m uvicorn backend.main:app --reload`. Swagger is available at `http://127.0.0.1:8000/docs`; the versioned health endpoint is `http://127.0.0.1:8000/api/v1/health`. Streamlit calls the API for price recommendations. See [backend/README.md](backend/README.md) for request examples and Render deployment instructions.

For a single public user link, deploy `app.py` on Streamlit Community Cloud. When `PRICING_API_URL` is not set, Streamlit uses the saved model and shared pricing engine locally inside the same service. Set `PRICING_API_URL` only when you want the frontend to call a separately deployed FastAPI backend.

## Deployment

Push the project to GitHub, commit the supplied CSV and generated model artifacts, create a Streamlit Community Cloud app, select `app.py`, and let Streamlit install the lightweight `requirements.txt`. TensorFlow is in `requirements-dl.txt` only and is not required for deployment.

## Limitations And Future Work

Historical data may not represent future behavior. The dataset ends in 2024, competitors can change prices, customer behavior can shift, and predicted demand is uncertain. Future work could add continuously updated POS data, live competitor collection, customer segments, A/B tests, online learning, and more formal forecasting. Reinforcement learning should only be considered after reliable real-world experimentation data exists.

## Presentation Structure

For a 5-7 minute presentation: business problem; why pricing is difficult; data and EDA findings; ML demand prediction; optional DL comparison; price simulation demo; metrics and business impact; limitations; future work.

Likely technical questions: Why predict demand instead of price? Why avoid leakage? Why compare multiple models? Why can Random Forest outperform Linear Regression? What do MAE and RMSE mean? Why is DL optional? How are categorical fields encoded? How are constraints enforced? How is reproducibility achieved? Why is this not autonomous pricing?

Likely business questions: What happens when cost rises? How does promotion affect a recommendation? Can this guarantee profit? What if competitor data is stale? Who approves the price? How often should the model be retrained? What if inventory is low? Can the same logic work across categories? What data would improve it? How would you measure business impact? The short answers are: the engine raises the floor; promotion is a demand input; no guarantee; stale data is a limitation; a human reviews it; retrain as data changes; inventory caps demand; category is included; real transaction history helps; and impact should be tested with controlled price experiments.