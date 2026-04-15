"""
Pydantic schemas for the Pre-Sowing Advisor API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class BeforeSowingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in percentage")
    rainfall: float = Field(..., description="Rainfall in mm")
    ph: float = Field(..., description="Soil pH value")
    soil_type: str = Field(..., description="Type of soil (e.g., clay, sandy, loamy, silt)")
    season: str = Field(..., description="Season (e.g., Kharif, Rabi, Zaid)")
    state_name: str = Field(..., description="Name of the state", alias="state")
    district_name: str = Field(..., description="Name of the district", alias="district")
    area: Optional[float] = Field(default=None, description="Land area in hectares (optional)")
    model_mode: Optional[str] = Field(
        default="standard",
        description="Inference mode: 'standard' or 'federated'.",
    )

    # Accept both old (state_name/district_name) and new (state/district) keys
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("model_mode")
    @classmethod
    def validate_model_mode(cls, v: Optional[str]) -> str:
        allowed = {"standard", "federated"}
        value = "standard" if v is None else str(v).strip().lower()
        if value not in allowed:
            raise ValueError(f"model_mode must be one of {allowed}. Got: '{v}'")
        return value


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------

class CropPrediction(BaseModel):
    crop: str
    confidence: float


class CropRecommenderResult(BaseModel):
    top_3: List[CropPrediction]
    selected_crop: str
    selected_confidence: float


class ConfidenceBand(BaseModel):
    lower: float
    upper: float


class YieldPredictorResult(BaseModel):
    expected_yield: float
    unit: str = "t/ha"
    confidence_band: ConfidenceBand
    explanation: str


class AgriConditionResult(BaseModel):
    sunlight_hours: float
    irrigation_type: str
    irrigation_need: str
    explanation: str = ""
    district_prior_used: bool = False
    district_irrigation_summary: str = ""
    irrigation_reasoning: str = ""
    irrigation_type_probabilities: Dict[str, float] = {}


class DistrictIntelligenceResult(BaseModel):
    district_crop_share_percent: Optional[float] = None
    yield_trend: str = ""
    top_competing_crops: List[str] = []
    best_historical_season: str = ""
    ten_year_trajectory_summary: str = ""
    irrigation_infrastructure_summary: str = ""
    irrigation_infrastructure_breakdown: Dict[str, float] = {}
    crop_irrigated_area_percent: Optional[float] = None
    insights: List[str] = []


# ---------------------------------------------------------------------------
# Full Response
# ---------------------------------------------------------------------------

class BeforeSowingResponse(BaseModel):
    success: bool = True
    input_summary: Dict[str, Any] = {}
    crop_recommender: CropRecommenderResult
    yield_predictor: YieldPredictorResult
    agri_condition_advisor: AgriConditionResult
    district_intelligence: DistrictIntelligenceResult
    system_notes: List[str] = []

    # Legacy backward-compat keys
    recommended_crop: str = ""
    predicted_yield: float = 0.0
    expected_yield: float = 0.0
    sunlight_hours: float = 0.0
    irrigation_type: str = ""
    irrigation_need: str = ""
    confidence: float = 0.0
    top_3_predictions: List[CropPrediction] = []
    model_type: str = "standard"
