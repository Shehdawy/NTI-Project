"""FastAPI application for AI retail price recommendations."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import DemandRequest, DemandResponse, ErrorResponse, HealthResponse, ModelInfoResponse, RecommendationRequest, RecommendationResponse, Recommendation, Scenario
from backend.services.model_service import ModelService
from backend.services.prediction_service import predict_demand
from backend.services.pricing_service import simulate


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_SERVICE = ModelService(ROOT)
configured_model_path = os.getenv("MODEL_PATH")
if configured_model_path:
    MODEL_SERVICE.model_path = (ROOT / configured_model_path).resolve()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_SERVICE.model_path.exists():
        MODEL_SERVICE.load()
    yield


origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:8502,http://localhost:8503").split(",") if origin.strip()]
app = FastAPI(title="AI Retail Pricing API", version="1.0.0", description="Demand prediction and price simulation using the existing trained retail model.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unexpected_error(_: Request, __: Exception):
    return JSONResponse(status_code=500, content={"error": {"code": "PREDICTION_ERROR", "message": "The prediction service could not complete the request."}})


@app.get("/", tags=["General"])
def root():
    return {"name": "AI Retail Pricing API", "version": "1.0.0", "status": "running", "description": "AI-powered retail demand and price recommendation API"}


@app.get("/api/v1/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(status="healthy" if MODEL_SERVICE.loaded else "degraded", model_loaded=MODEL_SERVICE.loaded)


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if not MODEL_SERVICE.loaded:
        return JSONResponse(status_code=503, content={"error": {"code": "MODEL_UNAVAILABLE", "message": "The demand model is not loaded."}})
    return ModelInfoResponse(model_type=MODEL_SERVICE.metadata.get("best_model", "Saved demand model"), task="Demand Prediction", features=MODEL_SERVICE.metadata.get("features", []), metrics=MODEL_SERVICE.metrics)


@app.post("/api/v1/predict-demand", response_model=DemandResponse, responses={422: {"model": ErrorResponse}}, tags=["Prediction"])
def predict(request: DemandRequest):
    try:
        return DemandResponse(predicted_demand=round(predict_demand(MODEL_SERVICE, request), 4))
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT", "message": str(error)}) from error


@app.post("/api/v1/recommend-price", response_model=RecommendationResponse, responses={422: {"model": ErrorResponse}}, tags=["Pricing"])
def recommend(request: RecommendationRequest):
    try:
        result, simulation = simulate(MODEL_SERVICE, request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_REQUEST", "message": str(error)}) from error
    scenarios = [Scenario(price=float(row.candidate_price), predicted_demand=float(row.predicted_demand), expected_revenue=float(row.expected_revenue), expected_profit=float(row.expected_profit) if row.expected_profit == row.expected_profit else None) for row in simulation.itertuples()]
    return RecommendationResponse(
        recommendation=Recommendation(recommended_price=result["recommended_price"], predicted_demand=result["predicted_demand"], expected_revenue=result["expected_revenue"], expected_profit=result["expected_profit"], profit_margin=result["profit_margin"]),
        current_price=request.current_price,
        objective=request.objective,
        scenarios=scenarios,
        user_provided_assumption=True,
        explanation=["The recommended price produced the best expected outcome among the tested scenarios.", f"Demand is predicted using the saved {MODEL_SERVICE.metadata.get('best_model', 'demand')} model.", "Cost, competitor price, and inventory are request inputs for this scenario; they are not inferred from the API request."],
    )