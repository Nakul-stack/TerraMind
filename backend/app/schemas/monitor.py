from pydantic import BaseModel, Field

class GrowthStageRequest(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in percentage")
    moisture: float = Field(..., description="Soil moisture in percentage")
    soil_type: str = Field(..., description="Type of soil (e.g., clay, sandy, loamy, silt)")
    crop_type: str = Field(..., description="Type of crop")
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    ph: float = Field(..., description="Soil pH value")
    rainfall: float = Field(..., description="Rainfall in mm")

class GrowthStageResponse(BaseModel):
    recommended_fertilizer: str = Field(..., description="Recommended fertilizer type")
    pest_level: str = Field(..., description="Predicted pest level severity")
    dosage: float = Field(..., description="Recommended dosage of fertilizer")
    apply_after_days: int = Field(..., description="Days after which to apply fertilizer")
    expected_yield_after_dosage: float = Field(..., description="Expected yield after applying the dosage")
