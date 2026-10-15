"""Stochastic (s, S) inventory replenishment policy solver for retail supply chains.

Translates conditional demand distributions from the forecasting model into optimal
operational inventory parameters (s = Reorder Point, S = Order-Up-To Level).

Mathematical formulation:
-------------------------
Let:
    - L: Supplier lead time (days)
    - R: Review period interval (days)
    - D_{L+R}: Aggregate demand during protection interval (L + R)
    - mu_{L+R} = E[D_{L+R}], sigma_{L+R} = Std(D_{L+R})
    - alpha: Target Cycle Service Level (CSL, e.g. 0.95)
    - z_alpha = Phi^-1(alpha): Standard normal quantile for service level
    - SS = z_alpha * sigma_{L+R}: Safety Stock
    - s = mu_{L+R} + SS: Reorder Point
    - Q* = sqrt(2 * K * mu / h): Economic Order Quantity (EOQ)
    - S = s + Q*: Order-Up-To Level

Perishable Constraint (Spoilage Mitigation):
--------------------------------------------
For perishable goods (shelf-life shelf_life_days), an inventory position exceeding
shelf_life_days * mu causes guaranteed waste (merma). The upper ceiling is enforced:
    S_effective = min(S, shelf_life_days * mu_daily)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class InventoryParameters:
    """Operational parameters defining an (s, S) inventory policy."""

    reorder_point_s: float
    order_up_to_S: float
    safety_stock: float
    expected_order_quantity: float
    service_level_target: float
    is_perishable: bool


class InventoryPolicySolver:
    """Calculates stochastic (s, S) inventory replenishment thresholds."""

    def __init__(
        self,
        lead_time_days: int = 3,
        review_period_days: int = 1,
        service_level_target: float = 0.95,
        ordering_cost_fixed: float = 50.0,
        holding_cost_unit_daily: float = 0.05,
        default_perishable_shelf_life: int = 7,
    ) -> None:
        self.lead_time = lead_time_days
        self.review_period = review_period_days
        self.service_level = service_level_target
        self.ordering_cost = ordering_cost_fixed
        self.holding_cost = holding_cost_unit_daily
        self.perishable_shelf_life = default_perishable_shelf_life

    def solve(
        self,
        daily_forecast_mean: float,
        daily_forecast_std: float,
        is_perishable: bool = False,
        shelf_life_days: int | None = None,
    ) -> InventoryParameters:
        """Solve for optimal reorder point (s) and order-up-to level (S).

        Parameters
        ----------
        daily_forecast_mean : float
            Expected daily sales (mu). Must be >= 0.
        daily_forecast_std : float
            Standard deviation of daily sales (sigma). Must be >= 0.
        is_perishable : bool, default=False
            Whether the item is perishable (receives spoilage ceiling).
        shelf_life_days : int | None, default=None
            Usable shelf-life in days before spoilage occurs.
        """
        mu = max(float(daily_forecast_mean), 0.0)
        sigma = max(float(daily_forecast_std), 0.0)

        # Protection interval: L + R days
        protection_interval = self.lead_time + self.review_period
        mu_protection = mu * protection_interval
        # Assuming independent daily errors: Var(sum) = sum(Var) -> sigma_total = sigma * sqrt(L + R)
        sigma_protection = sigma * math.sqrt(protection_interval)

        # Standard normal quantile: z_alpha = Phi^-1(alpha)
        z_score = float(norm.ppf(self.service_level))
        safety_stock = z_score * sigma_protection

        # Reorder point s:
        s = mu_protection + safety_stock

        # Economic order quantity Q:
        # EOQ = sqrt(2 * K * annual_demand / annual_holding) = sqrt(2 * K * mu / h_daily)
        if mu > 0.0 and self.holding_cost > 0.0:
            q_star = math.sqrt((2.0 * self.ordering_cost * mu) / self.holding_cost)
        else:
            q_star = max(mu * self.lead_time, 1.0)

        # Order-up-to level S:
        S = s + q_star

        # Spoilage constraint for perishable items:
        # Do not allow S to exceed what can be sold within the remaining shelf life
        effective_shelf_life = shelf_life_days or self.perishable_shelf_life
        if is_perishable and mu > 0.0:
            max_perishable_cap = mu * effective_shelf_life
            if max_perishable_cap < S:
                S = max(max_perishable_cap, s + 1.0)

        return InventoryParameters(
            reorder_point_s=round(s, 2),
            order_up_to_S=round(S, 2),
            safety_stock=round(safety_stock, 2),
            expected_order_quantity=round(q_star, 2),
            service_level_target=self.service_level,
            is_perishable=is_perishable,
        )
