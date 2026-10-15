"""Unit tests for the inventory policy solver and Monte Carlo simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.inventory_policy import InventoryPolicySolver
from src.optimization.simulation import MonteCarloInventorySimulator


class TestInventoryPolicySolver:
    """Test suite for (s, S) inventory policy solver."""

    def test_reorder_point_strictly_less_than_order_up_to(self) -> None:
        """In any valid policy, s must be strictly less than S."""
        solver = InventoryPolicySolver(
            lead_time_days=3, review_period_days=1, service_level_target=0.95
        )
        params = solver.solve(
            daily_forecast_mean=100.0, daily_forecast_std=20.0, is_perishable=False
        )

        assert params.reorder_point_s < params.order_up_to_S
        assert params.safety_stock > 0.0
        assert params.expected_order_quantity > 0.0

    def test_service_level_sensitivity(self) -> None:
        """Higher target service level must yield higher safety stock and higher reorder point."""
        solver_low = InventoryPolicySolver(service_level_target=0.90)
        solver_high = InventoryPolicySolver(service_level_target=0.99)

        mean_demand = 80.0
        std_demand = 25.0

        policy_low = solver_low.solve(mean_demand, std_demand)
        policy_high = solver_high.solve(mean_demand, std_demand)

        assert policy_high.safety_stock > policy_low.safety_stock
        assert policy_high.reorder_point_s > policy_low.reorder_point_s

    def test_perishable_shelf_life_cap(self) -> None:
        """For perishable goods with tight shelf life, S must be capped to prevent guaranteed spoilage."""
        # High fixed ordering cost pushes EOQ high, but shelf life is short (3 days)
        solver = InventoryPolicySolver(
            lead_time_days=1,
            review_period_days=1,
            ordering_cost_fixed=500.0,
            holding_cost_unit_daily=0.01,
        )
        mean_demand = 20.0
        std_demand = 2.0

        # Non-perishable will have very large S due to EOQ
        non_perishable_policy = solver.solve(mean_demand, std_demand, is_perishable=False)

        # Perishable with 3 days shelf life
        perishable_policy = solver.solve(
            mean_demand, std_demand, is_perishable=True, shelf_life_days=3
        )

        assert perishable_policy.order_up_to_S < non_perishable_policy.order_up_to_S


class TestMonteCarloInventorySimulator:
    """Test suite for Monte Carlo discrete-event inventory simulation."""

    def test_simulation_runs_and_produces_coherent_kpis(self) -> None:
        """Verify KPI bounds: fill rate between 0 and 1, stockout frequency between 0 and 1."""
        sim = MonteCarloInventorySimulator(lead_time_days=2)
        horizon = 16
        means = np.full(horizon, 50.0)
        stds = np.full(horizon, 10.0)

        # Use an ample policy: s=180, S=350
        kpis = sim.simulate(
            reorder_point_s=180.0,
            order_up_to_S=350.0,
            forecast_means=means,
            forecast_stds=stds,
            n_mc_paths=100,
            seed=42,
        )

        assert 0.0 <= kpis.fill_rate <= 1.0
        assert 0.0 <= kpis.stockout_frequency <= 1.0
        assert kpis.fill_rate + kpis.stockout_frequency == pytest.approx(1.0, abs=1e-4)
        assert kpis.average_on_hand_inventory >= 0.0
        assert kpis.average_daily_cost >= 0.0

    def test_perishable_produces_spoilage(self) -> None:
        """When an item is perishable and daily spoilage rate > 0, spoilage units must be > 0."""
        sim = MonteCarloInventorySimulator(lead_time_days=2, daily_spoilage_rate=0.05)
        horizon = 16
        means = np.full(horizon, 30.0)
        stds = np.full(horizon, 5.0)

        kpis = sim.simulate(
            reorder_point_s=100.0,
            order_up_to_S=250.0,
            forecast_means=means,
            forecast_stds=stds,
            n_mc_paths=100,
            is_perishable=True,
            seed=42,
        )

        assert kpis.total_spoilage_units > 0.0
        assert kpis.spoilage_cost_total > 0.0

    def test_empty_horizon_raises_error(self) -> None:
        """Empty forecast horizon array must raise ValueError."""
        sim = MonteCarloInventorySimulator()
        with pytest.raises(ValueError, match="must be > 0"):
            sim.simulate(10.0, 20.0, np.array([]), np.array([]))
