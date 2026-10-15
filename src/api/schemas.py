"""Pydantic v2 request/response schemas for the forecast API.

TODO: Implement in the next development iteration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request schema for the /forecast endpoint."""

    store_nbr: int = Field(..., ge=1, le=54, description="Store number (1-54).")
    family: str = Field(..., description="Product family (e.g., 'GROCERY I').")
    horizon_days: int = Field(default=16, ge=1, le=30, description="Forecast horizon in days.")


class ForecastResponse(BaseModel):
    """Response schema for the /forecast endpoint."""

    store_nbr: int
    family: str
    predictions: list[float] = Field(..., description="Daily predicted unit sales.")
    prediction_lower: list[float] = Field(..., description="Lower bound (5th percentile).")
    prediction_upper: list[float] = Field(..., description="Upper bound (95th percentile).")


class InventoryPolicyRequest(BaseModel):
    """Request schema for the /optimize endpoint."""

    store_nbr: int = Field(..., ge=1, le=54)
    family: str
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)


class InventoryPolicyResponse(BaseModel):
    """Response schema for the /optimize endpoint."""

    store_nbr: int
    family: str
    reorder_point_s: float = Field(..., description="Reorder point (s).")
    order_up_to_S: float = Field(..., description="Order-up-to level (S).")
    expected_fill_rate: float
    expected_daily_holding_cost: float
    expected_daily_stockout_rate: float
