# PricePilot AI — Retail Pricing API

FastAPI backend for the existing retail demand model and price simulation engine. It does not retrain or duplicate the ML model. The API loads `models/demand_model.pkl` once at startup and calls the existing `src.pricing_engine.recommend_price` function for recommendations.

## Architecture

```text
User -> Streamlit -> FastAPI -> validation -> saved ML model -> pricing engine -> JSON response
```

## Endpoints

- `GET /` API information
- `GET /api/v1/health` model health
- `GET /api/v1/model/info` saved model features and metrics
- `POST /api/v1/predict-demand` demand estimate
- `POST /api/v1/recommend-price` candidate-price simulation and recommendation
- `/docs` Swagger UI
- `/redoc` ReDoc

The request fields match the actual trained model: `product_category`, `date`, `price` or `current_price`, `cost_price`, `competitor_price`, `inventory`, `promotion`, `season`, and `customer_rating`. There is no `store` or `historical_demand` feature in the saved model, so those fields are not accepted. Cost, competitor price, inventory, and other scenario values are supplied inputs; the API does not claim they came from a particular historical row.

## Run locally

From the project root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`.

Example recommendation:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/recommend-price -H "Content-Type: application/json" -d "{\"product_category\":\"Home\",\"date\":\"2024-12-31\",\"current_price\":164.36,\"cost_price\":86.77,\"competitor_price\":149.54,\"inventory\":101,\"promotion\":true,\"season\":\"Winter\",\"customer_rating\":4.4,\"objective\":\"maximize_revenue\"}"
```

## Configuration and CORS

Copy `backend/.env.example` to `.env` and set `ALLOWED_ORIGINS` as a comma-separated list. In production, set it to the deployed Streamlit URL. No secrets are required.

## Deployment on Render

1. Push the repository to GitHub.
2. Create a Render Web Service from the repository.
3. Set the build command to `pip install -r requirements.txt -r backend/requirements.txt`.
4. Set the start command to `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
5. Set `ALLOWED_ORIGINS` to the Streamlit deployment URL.
6. Verify `/api/v1/health`, then open `/docs`.

## Limitations

Recommendations are estimates based on historical data ending in 2024. They are not guarantees of current sales, revenue, or profit. The API is a decision-support service and has no authentication, rate limiting, or database.