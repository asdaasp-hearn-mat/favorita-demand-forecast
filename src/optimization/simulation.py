"""Monte Carlo stochastic inventory simulation engine.

Simulates discrete-event daily inventory dynamics under an (s, S) policy across
hundreds of stochastic demand trajectories to quantify the trade-off between
stockouts (quiebres) and perishable spoilage (mermas).

State variables:
----------------
- I_t: On-hand inventory at dawn of day t
- Pipeline: Orders currently in transit from supplier (lead time L)
- D_t: Realized stochastic demand ~ D(mu_t, sigma_t)
- Fulfilled_t = min(I_t + Arrivals_t, D_t)
- Stockout_t = max(D_t - (I_t + Arrivals_t), 0)
- Spoilage_t = theta * max(I_t + Arrivals_t - Fulfilled_t, 0)
- Ordering decision: if Inventory_Position <= s, order S - Inventory_Position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class SimulationKPIs:
    """Aggregated performance metrics from Monte Carlo simulation runs."""

    fill_rate: float
    stockout_frequency: float
    average_on_hand_inventory: float
    total_spoilage_units: float
    total_stockout_units: float
    average_daily_cost: float
    holding_cost_total: float
    stockout_penalty_total: float
    spoilage_cost_total: float
    ordering_cost_total: float


class MonteCarloInventorySimulator:
    """Simulates inventory trajectories under stochastic demand and lead time."""

    def __init__(
        self,
        lead_time_days: int = 3,
        holding_cost_per_unit_day: float = 0.05,
        stockout_penalty_per_unit: float = 2.50,
        spoilage_cost_per_unit: float = 1.50,
        ordering_cost_fixed: float = 25.0,
        daily_spoilage_rate: float = 0.04,
    ) -> None:
        self.lead_time = lead_time_days
        self.holding_cost = holding_cost_per_unit_day
        self.stockout_penalty = stockout_penalty_per_unit
        self.spoilage_cost = spoilage_cost_per_unit
        self.ordering_cost = ordering_cost_fixed
        self.daily_spoilage_rate = daily_spoilage_rate

    def simulate(
        self,
        reorder_point_s: float,
        order_up_to_S: float,
        forecast_means: npt.NDArray[np.float64],
        forecast_stds: npt.NDArray[np.float64],
        n_mc_paths: int = 1000,
        initial_inventory: float | None = None,
        is_perishable: bool = False,
        seed: int = 42,
    ) -> SimulationKPIs:
        """Execute Monte Carlo simulation across multiple stochastic demand realizations.

        Parameters
        ----------
        reorder_point_s : float
            Threshold s that triggers an order.
        order_up_to_S : float
            Target inventory level S.
        forecast_means : NDArray[float64]
            Vector of predicted daily demand means over simulation horizon (length T).
        forecast_stds : NDArray[float64]
            Vector of predicted daily demand standard deviations (length T).
        n_mc_paths : int, default=1000
            Number of Monte Carlo iterations.
        initial_inventory : float | None
            Starting on-hand stock. If None, defaults to order_up_to_S.
        is_perishable : bool, default=False
            Whether to apply daily spoilage decay to unsold stock.
        seed : int, default=42
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        T = len(forecast_means)
        if T == 0:
            raise ValueError("Forecast horizon length must be > 0.")

        init_inv = float(order_up_to_S) if initial_inventory is None else float(initial_inventory)
        spoilage_rate = self.daily_spoilage_rate if is_perishable else 0.0

        # Pre-allocate trajectory accumulators
        total_demand_all = 0.0
        total_fulfilled_all = 0.0
        total_stockout_all = 0.0
        total_spoilage_all = 0.0
        total_holding_all = 0.0
        total_orders_placed_all = 0

        # Sample demand paths: matrix shape (n_mc_paths, T)
        # Using truncated normal (non-negative demands)
        mu_matrix = np.tile(forecast_means, (n_mc_paths, 1))
        sigma_matrix = np.tile(forecast_stds, (n_mc_paths, 1))
        demands = rng.normal(mu_matrix, sigma_matrix)
        demands = np.clip(demands, a_min=0.0, a_max=None)

        for p in range(n_mc_paths):
            on_hand = init_inv
            # Pipeline tracks orders scheduled to arrive at day t: dict[arrival_day, quantity]
            pipeline: list[float] = [0.0] * (T + self.lead_time + 1)
            orders_count = 0

            for t in range(T):
                d_t = float(demands[p, t])
                total_demand_all += d_t

                # 1. Pipeline arrivals
                arrivals = pipeline[t]
                on_hand += arrivals

                # 2. Demand fulfillment
                fulfilled = min(on_hand, d_t)
                stockout = max(d_t - on_hand, 0.0)
                on_hand -= fulfilled

                total_fulfilled_all += fulfilled
                total_stockout_all += stockout

                # 3. Spoilage evaluation on residual stock
                spoilage = on_hand * spoilage_rate
                on_hand -= spoilage
                total_spoilage_all += spoilage

                # 4. Holding inventory
                total_holding_all += on_hand

                # 5. Review & Order decision (Inventory position = on-hand + in-transit)
                in_transit = sum(pipeline[t + 1 : t + 1 + self.lead_time])
                inv_pos = on_hand + in_transit

                if inv_pos <= reorder_point_s:
                    q_order = max(order_up_to_S - inv_pos, 0.0)
                    if q_order > 0:
                        arrival_day = t + self.lead_time
                        if arrival_day < len(pipeline):
                            pipeline[arrival_day] += q_order
                        orders_count += 1

            total_orders_placed_all += orders_count

        # Compute averages across all paths and days
        fill_rate = total_fulfilled_all / max(total_demand_all, 1e-9)
        stockout_freq = total_stockout_all / max(total_demand_all, 1e-9)
        avg_on_hand = total_holding_all / (n_mc_paths * T)
        avg_spoilage = total_spoilage_all / n_mc_paths
        avg_stockout = total_stockout_all / n_mc_paths

        # Cost accounting
        holding_cost = total_holding_all * self.holding_cost / n_mc_paths
        stockout_cost = total_stockout_all * self.stockout_penalty / n_mc_paths
        spoilage_cost = total_spoilage_all * self.spoilage_cost / n_mc_paths
        ordering_cost = total_orders_placed_all * self.ordering_cost / n_mc_paths

        total_cost = holding_cost + stockout_cost + spoilage_cost + ordering_cost
        avg_daily_cost = total_cost / T

        return SimulationKPIs(
            fill_rate=round(float(fill_rate), 4),
            stockout_frequency=round(float(stockout_freq), 4),
            average_on_hand_inventory=round(float(avg_on_hand), 2),
            total_spoilage_units=round(float(avg_spoilage), 2),
            total_stockout_units=round(float(avg_stockout), 2),
            average_daily_cost=round(float(avg_daily_cost), 2),
            holding_cost_total=round(float(holding_cost), 2),
            stockout_penalty_total=round(float(stockout_cost), 2),
            spoilage_cost_total=round(float(spoilage_cost), 2),
            ordering_cost_total=round(float(ordering_cost), 2),
        )


if __name__ == "__main__":
    import argparse

    import yaml

    from src.optimization.inventory_policy import InventoryPolicySolver

    parser = argparse.ArgumentParser(description="Run inventory policy Monte Carlo simulation.")
    parser.add_argument(
        "--config", type=str, default="configs/base_config.yaml", help="Path to YAML config."
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inv_cfg = cfg.get("inventory", {})
    lead_time = inv_cfg.get("lead_time_days", 3)
    service_level = inv_cfg.get("service_level_target", 0.95)
    n_paths = inv_cfg.get("simulation_mc_paths", 1000)
    horizon = cfg.get("forecast", {}).get("horizon_days", 16)

    print(f"--- Running Monte Carlo (s, S) Inventory Optimization ({n_paths} paths) ---")
    solver = InventoryPolicySolver(
        service_level_target=service_level,
        lead_time_days=lead_time,
        review_period_days=inv_cfg.get("review_period_days", 1),
    )
    policy = solver.solve(daily_forecast_mean=85.0, daily_forecast_std=20.0, is_perishable=True)
    print(
        f"Optimal Policy (Perishable): s={policy.reorder_point_s}, S={policy.order_up_to_S}, "
        f"Safety Stock={policy.safety_stock}"
    )

    simulator = MonteCarloInventorySimulator(lead_time_days=lead_time)
    kpis = simulator.simulate(
        reorder_point_s=policy.reorder_point_s,
        order_up_to_S=policy.order_up_to_S,
        forecast_means=np.full(horizon, 85.0),
        forecast_stds=np.full(horizon, 20.0),
        n_mc_paths=n_paths,
        is_perishable=True,
    )
    print("Simulation Performance KPIs:")
    print(f"  Fill Rate: {kpis.fill_rate * 100:.2f}% (Target: {service_level * 100:.1f}%)")
    print(f"  Stockout Frequency: {kpis.stockout_frequency * 100:.2f}%")
    print(f"  Avg On-Hand: {kpis.average_on_hand_inventory} units")
    print(f"  Avg Spoilage: {kpis.total_spoilage_units} units")
    print(f"  Avg Daily Total Cost: ${kpis.average_daily_cost:.2f}")
