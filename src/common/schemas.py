from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    vendor_id: int = Field(..., ge=0)
    passenger_count: int = Field(..., ge=0)
    trip_distance: float = Field(..., ge=0)
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_weekday: int = Field(..., ge=0, le=6)
    rate_code: int = Field(..., ge=0)
    payment_type: int = Field(..., ge=0)

class PredictResponse(BaseModel):
    predicted_duration_min: float
