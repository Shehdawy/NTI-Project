"""Validated API request and response contracts."""

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PricingObjective(str, Enum):
    REVENUE = "maximize_revenue"
    PROFIT = "maximize_profit"
    BALANCED = "balanced"


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"product_category": "HOBBIES", "date": "2016-05-22", "price": 9.58, "cost_price": 6.23, "competitor_price": 9.87, "inventory": 101, "promotion": True, "season": "Spring", "customer_rating": 4.1}})

    product_category: str = Field(min_length=1, description="Product category present in the supplied data.")
    date: date
    cost_price: float = Field(gt=0)
    competitor_price: float = Field(gt=0)
    inventory: int = Field(gt=0)
    promotion: bool = False
    season: str = Field(min_length=1)
    customer_rating: float = Field(ge=1, le=5)

    @field_validator("product_category", "season")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class DemandRequest(ScenarioRequest):
    price: float = Field(gt=0)


class RecommendationRequest(ScenarioRequest):
    current_price: float = Field(gt=0, description="Current selling price used as the simulation center.")
    objective: PricingObjective = PricingObjective.REVENUE


class DemandResponse(BaseModel):
    predicted_demand: float
    unit: str = "units"
    confidence_note: str = "Prediction is an estimate based on historical data."


class Scenario(BaseModel):
    price: float
    predicted_demand: float
    expected_revenue: float
    expected_profit: float | None = None


class Recommendation(BaseModel):
    recommended_price: float
    predicted_demand: float
    expected_revenue: float
    expected_profit: float | None = None
    profit_margin: float | None = None


class RecommendationResponse(BaseModel):
    recommendation: Recommendation
    current_price: float
    objective: PricingObjective
    scenarios: list[Scenario]
    explanation: list[str]
    user_provided_assumption: bool = False
    warning: str = "This recommendation is an estimate and not a guarantee of future sales."


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    task: str
    features: list[str]
    metrics: dict[str, float]