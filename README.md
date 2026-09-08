# AI-Based Retail Price Recommendation System

## 1. Business Problem

Retailers must balance demand and margin. A price that is too high can reduce sales, while a price that is too low can reduce revenue and gross profit. This project provides a repeatable decision-support workflow for reviewing candidate prices.

## 2. Proposed Solution

The system has two separate responsibilities:

- **Prediction:** the ML model predicts expected units sold from historical retail data.
- **Decision logic:** the pricing engine simulates candidate prices and selects a recommended price according to a business objective.

The system does not claim to predict a perfect price or current Walmart prices.

## 3. Dataset

The project uses the historical M5 Walmart retail forecasting dataset. It does not represent current 2026 market conditions. Download these files from the official M5 competition source and place them in `data/raw/`:

- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `calendar.csv`

The raw CSV files are ignored by Git because they are too large for a normal repository. The loader uses a deterministic product sample and the most recent available evaluation period.

M5 does not provide true cost, competitor price, inventory, or customer rating. The project therefore uses documented scenario assumptions for the prepared data and UI defaults:

- Cost price: 65% of selling price
- Competitor price: 103% of selling price
- Inventory: 100 units
- Customer rating: 4.1

These assumptions are used for pricing calculations and are not learned demand features.

## 4. EDA and Data Preparation

`model_training.py` loads and reshapes the M5 files, cleans invalid rows, creates the prepared dataset, and generates a basic EDA image in `reports/`.

The prepared data contains date, category, selling price, promotion, season, weekday, observed units sold, and derived pricing fields. Missing required values are removed before the chronological split.

## 5. Machine Learning

The target is `units_sold`, interpreted as an observed-sales demand proxy. The trained features are:

- Product category
- Selling price
- Promotion
- Season
- Day of week
- Month
- Weekend flag

The project compares:

- Mean-demand baseline
- Linear Regression
- Random Forest
- Gradient Boosting

The final holdout is chronological and split at a date boundary so records from the same day cannot appear in both train and test. The selected model is serialized to `models/demand_model.joblib`.

## 6. Deep Learning

`src/deep_learning_model.py` contains an optional small Keras network for comparison. TensorFlow is not required by the deployed application.

## 7. Pricing Simulation

For each request, the pricing engine tests candidate prices within current-price and competitor-aware limits. For each candidate it:

1. Predicts expected units sold.
2. Caps expected units by the requested inventory.
3. Calculates expected revenue.
4. Calculates expected gross profit using the supplied cost assumption.
5. Calculates a normalized balanced score when requested.

The model predicts units sold; the pricing engine makes the price decision.

## 8. Business Objectives

The user can select:

- **Maximize revenue**
- **Maximize profit**
- **Balanced**, using normalized revenue and profit scores

These are scenario estimates, not guarantees of future sales.

## 9. FastAPI

The API is implemented in `backend/` and exposes:

- `GET /`
- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/predict-demand`
- `POST /api/v1/recommend-price`

Interactive API documentation is available at `/docs` when FastAPI is running.

## 10. Streamlit

`streamlit_app.py` provides the interactive decision dashboard. It can run in two modes:

- Local mode: loads the prepared dataset and model directly.
- API mode: set `PRICING_API_URL` to the FastAPI base URL, such as `http://127.0.0.1:8000`.

## 11. Project Structure

```text
streamlit_app.py               Streamlit dashboard
model_training.py              Training and evaluation entry point
README.md                      Project and presentation documentation
backend/                       FastAPI application and services
src/                           Data preparation, pricing, and optional deep learning code
data/raw/                     Downloaded M5 files, ignored by Git
data/processed/               Prepared local data, ignored by Git
models/                        Model instructions and ignored artifacts
reports/                       Regenerated EDA and evaluation outputs
tests/                         API tests
```

## 12. Installation

```powershell
python -m pip install -r requirements.txt
```

Place the three M5 files in `data/raw/`, then generate the prepared data and model:

```powershell
python model_training.py
```

## 13. Running the Project

Run the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

Run FastAPI separately:

```powershell
python -m uvicorn backend.main:app --reload
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

To use FastAPI from Streamlit, set `PRICING_API_URL=http://127.0.0.1:8000` before starting Streamlit.

## 14. Model Evaluation

The training script writes model comparisons to `models/model_metrics.csv` and test predictions to `reports/test_predictions.csv`.

- **MAE:** average absolute prediction error.
- **RMSE:** prediction error that penalizes larger errors more heavily.
- **R²:** the proportion of target variation explained by the model.

The current evaluation is a chronological holdout. Rolling time-series validation would be a useful future improvement.

## 15. Limitations

- M5 is historical and ends in 2016; it does not describe current 2026 Walmart prices.
- The target is observed units sold, not directly observed unmet demand.
- Cost, competitor price, inventory, and customer rating are assumptions.
- Price response is learned from observational historical prices, not a controlled pricing experiment.
- Recommendations are category-level and do not provide store-specific optimization.
- A human should review recommendations before using them in a real business.

## 16. Future Improvements

- Integrate live POS and inventory data.
- Add real competitor prices and product costs.
- Add product and store identifiers to the demand model.
- Use rolling time-series validation and controlled price experiments.
- Add uncertainty intervals and monitoring.
- Add an optional LLM business assistant for explaining recommendations. No LLM assistant is currently implemented.


ize and push the repository only when run by the project owner. This assistant does not push anything automatically.
