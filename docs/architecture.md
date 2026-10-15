# System Architecture

> **Hierarchical Retail Demand Forecasting & Inventory Optimization**
> Corporación Favorita — Store Sales Time Series Forecasting

---

## High-Level System Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│                                                                              │
│  Kaggle CSVs ──► Polars LazyFrame Ingest ──► DuckDB Staging ──► Parquet     │
│  (train, items, stores, oil, holidays, transactions)                        │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                                     │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│  │  temporal.py     │    │ aggregations.py  │    │  External features       │  │
│  │  ─────────────── │    │ ──────────────── │    │  ────────────────────    │  │
│  │  • Calendar      │    │  • Lag features  │    │  • Oil price (WTI)      │  │
│  │  • Cyclical enc. │    │    (≥16d safe)   │    │  • Holiday proximity    │  │
│  │  • Payday flags  │    │  • Rolling stats │    │  • Promotion history    │  │
│  │  • Weekend/EOM   │    │  • Expanding avg │    │                          │  │
│  │                   │    │  • WoW / MoM diff│    │                          │  │
│  └─────────────────┘    └─────────────────┘    └──────────────────────────┘  │
│                                                                              │
│                    ⚠ LEAKAGE GUARD: MIN_LAG ≥ 16 days                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Time-Series Cross-Validation (Expanding Window)                       │ │
│  │  ═══════════════════════════════════════════════════════════════════════│ │
│  │  Fold 1: Train [──────────] Val [────]                                 │ │
│  │  Fold 2: Train [──────────────] Val [────]                             │ │
│  │  Fold 3: Train [──────────────────] Val [────]                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐   │
│  │  LightGBM    │  │  CatBoost    │  │  Hierarchical Reconciliation    │   │
│  │  Forecaster  │  │  Forecaster  │  │  (Bottom-Up / MinT)             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────────────┘   │
│                                                                              │
│  Metric: NWRMSLE (w_perishable=1.25, w_non_perishable=1.00)                │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION LAYER                                         │
│                                                                              │
│  Point forecast + prediction intervals ──►  (s, S) Policy Solver            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     Monte Carlo Simulation                             │ │
│  │  ─────────────────────────────────────────────────────────────────────  │ │
│  │  For each (store, family):                                             │ │
│  │    1. Sample N demand paths from predictive distribution               │ │
│  │    2. Simulate inventory dynamics with lead time & review period       │ │
│  │    3. Compute KPIs: stockout rate, spoilage rate, holding cost         │ │
│  │    4. Optimize (s, S) to minimize total cost at target service level   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                           │
│                                                                              │
│  FastAPI Service (src/api/main.py)                                          │
│  ├── POST /forecast   → 16-day demand predictions + intervals               │
│  ├── POST /optimize   → (s, S) inventory policy recommendation              │
│  └── GET  /health     → Service health check                                │
│                                                                              │
│  Docker (docker/Dockerfile.api + docker-compose.yml)                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Leakage Prevention Architecture

Data leakage is the most critical risk in time-series forecasting.  This project
implements a **three-layer defense**:

| Layer | Mechanism | Where |
|:------|:----------|:------|
| **Compile-time** | `MIN_SAFE_LAG = 16` constant.  `build_lag_features()` raises `ValueError` if any requested lag offset < forecast horizon. | `src/features/aggregations.py` |
| **Runtime gate** | `validate_no_leakage()` inspects column names for offset violations and checks that the train partition contains no sentinel values from the test period. | `src/features/aggregations.py` |
| **Test suite** | `TestDataLeakagePrevention` class injects unique sentinel values into the test period and asserts they never appear in train-partition feature columns. | `tests/test_features.py` |

---

## Evaluation Metric: NWRMSLE

The official competition metric penalizes forecast errors on **perishable items
more heavily** than non-perishable items:

$$\text{NWRMSLE} = \sqrt{\frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}_i + 1) - \ln(y_i + 1) \right)^2}{\sum_{i=1}^n w_i}}$$

where $w_i = 1.25$ for perishable and $w_i = 1.00$ for non-perishable items.

Two implementations are provided:
- **NumPy** (`nwrmsle_numpy`): For evaluation loops and sklearn-compatible scoring.
- **Polars** (`nwrmsle_polars`): For large-scale evaluation directly in the data pipeline
  with zero-copy memory efficiency and multithreaded SIMD execution.

---

## Inventory Optimization: (s, S) Policy

The prescriptive layer translates demand forecasts into actionable inventory decisions:

- **s (reorder point)**: When inventory drops to this level, trigger a replenishment order.
- **S (order-up-to level)**: Order enough to bring inventory up to this level.
- **Optimization objective**: Minimize `holding_cost + stockout_penalty + spoilage_cost`
  subject to a target service level (default 95%).

The Monte Carlo simulation samples from the predictive distribution of demand to account
for forecast uncertainty, lead-time variability, and perishable spoilage dynamics.

---

## Technology Choices

| Component | Technology | Rationale |
|:----------|:-----------|:----------|
| **Data Processing** | Polars (LazyFrame) | Multithreaded, zero-copy, memory-mapped; 5-10× faster than Pandas on this dataset |
| **Ad-hoc Queries** | DuckDB | In-process OLAP for exploratory analysis |
| **Gradient Boosting** | LightGBM + CatBoost | State-of-the-art for tabular time-series with categorical features |
| **API Framework** | FastAPI + Pydantic | Async, auto-documented (OpenAPI), type-safe request/response validation |
| **Package Management** | uv + pyproject.toml | Deterministic lockfiles, 10-100× faster than pip |
| **Code Quality** | ruff + mypy (strict) | Unified linter/formatter + static type safety |
| **CI/CD** | GitHub Actions | Automated lint, typecheck, and test on every push/PR |
