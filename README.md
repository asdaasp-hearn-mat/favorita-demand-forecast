# Hierarchical Retail Demand Forecasting & Inventory Optimization

[![CI](https://github.com/asdaasp-hearn-mat/favorita-demand-forecast/actions/workflows/lint-test.yml/badge.svg)](.github/workflows/lint-test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/dataframe-Polars-orange)](https://pola.rs/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **End-to-end production ML system** that forecasts daily retail demand at the (store × product family) level and optimizes inventory replenishment policies using stochastic simulation — built on the Corporación Favorita dataset (Kaggle).

---

## What This Project Demonstrates

| Competency | How It's Applied |
|:-----------|:-----------------|
| **Time-Series Feature Engineering** | 30+ features: calendar, cyclical encodings, Ecuadorian economic calendar, holiday proximity, oil-price signals, and lag/rolling aggregations — all with strict data-leakage prevention |
| **High-Performance Data Processing** | Zero Pandas in critical paths.  All transformations use **Polars LazyFrames** (multithreaded SIMD) and **DuckDB** for memory-efficient processing |
| **ML Model Development** | LightGBM & CatBoost ensembles evaluated with a custom weighted metric (NWRMSLE) under time-series cross-validation |
| **Operations Research** | Monte Carlo simulation of stochastic *(s, S)* inventory policies with spoilage dynamics for perishable goods |
| **Production Engineering** | Modular Python package, FastAPI serving layer, Docker deployment, GitHub Actions CI, strict typing (`mypy`), and comprehensive `pytest` suite |

---

## Problem Statement

**Corporación Favorita**, Ecuador's largest grocery retailer, operates 54 stores selling items across 33 product families.  The challenge is twofold:

1. **Forecast** daily unit sales per (store, product family) for a 16-day horizon, optimizing the official **NWRMSLE** metric that penalizes perishable-item errors 25% more heavily.

2. **Prescribe** inventory reorder points and order-up-to levels *(s, S)* that minimize total cost (holding + stockout penalty + spoilage) at a 95% target service level.

$$\text{NWRMSLE} = \sqrt{\frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}_i + 1) - \ln(y_i + 1) \right)^2}{\sum_{i=1}^n w_i}} \quad \text{where} \quad w_i = \begin{cases} 1.25 & \text{perishable} \\ 1.00 & \text{otherwise} \end{cases}$$

---

## Repository Structure

```
favorita-demand-forecast/
│
├── .github/workflows/
│   └── lint-test.yml           # CI: ruff + mypy + pytest on Python 3.11/3.12
│
├── configs/
│   ├── base_config.yaml        # Project-wide settings: paths, horizon, inventory params
│   ├── features_config.yaml    # Feature engineering hyperparameters
│   └── model_config.yaml       # LightGBM/CatBoost hyperparameters & CV strategy
│
├── data/
│   ├── raw/                    # Original Kaggle CSVs (git-ignored)
│   ├── processed/              # Engineered Parquet files (git-ignored)
│   └── sample/                 # Small sample for CI tests (committed)
│
├── docker/
│   ├── Dockerfile.api          # Production inference container
│   └── docker-compose.yml      # Orchestration with mounted volumes
│
├── docs/
│   └── architecture.md         # System architecture & design decisions
│
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis (visualization only)
│
├── src/                        # Production Python package
│   ├── data/
│   │   ├── loader.py           # Polars/DuckDB data ingestion
│   │   └── clean.py            # Cleaning, type casting, null handling
│   │
│   ├── features/
│   │   ├── temporal.py         # Calendar, cyclical, holiday, oil features
│   │   └── aggregations.py     # Lag, rolling, expanding + leakage guards
│   │
│   ├── models/
│   │   ├── metrics.py          # Vectorized NWRMSLE (NumPy + Polars)
│   │   ├── train.py            # Training orchestrator with TS cross-validation
│   │   ├── lgbm_forecaster.py  # LightGBM model wrapper
│   │   └── hierarchical.py     # Bottom-up / MinT reconciliation
│   │
│   ├── optimization/
│   │   ├── inventory_policy.py # (s, S) policy solver
│   │   └── simulation.py       # Monte Carlo inventory simulation
│   │
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic request/response models
│
├── tests/
│   ├── test_metrics.py         # NWRMSLE correctness & boundary tests
│   ├── test_features.py        # Feature correctness & leakage prevention
│   └── test_data_pipeline.py   # Data pipeline integration tests
│
├── .gitignore
├── Makefile                    # Automation: setup, lint, test, train, serve
├── pyproject.toml              # Dependencies, ruff, mypy, pytest config
└── README.md                   # Project documentation
```

---

## Data Leakage Prevention

Data leakage is the **#1 silent failure mode** in time-series ML projects.  This repository implements a three-layer defense:

| Layer | Mechanism | Location |
|:------|:----------|:---------|
| **Static guard** | All lag offsets must be ≥ `MIN_SAFE_LAG` (16 days = forecast horizon).  `build_lag_features()` raises `ValueError` at construction time if violated. | [`aggregations.py`](src/features/aggregations.py) |
| **Runtime validator** | `validate_no_leakage()` inspects lag column names for unsafe offsets and checks feature provenance. | [`aggregations.py`](src/features/aggregations.py) |
| **Sentinel injection test** | Injects a unique sentinel value in the test period and asserts it **never** appears in train-partition feature columns. | [`test_features.py`](tests/test_features.py) |

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/asdaasp-hearn-mat/favorita-demand-forecast.git
cd favorita-demand-forecast

# Set up the environment (requires uv: https://docs.astral.sh/uv/)
make setup

# Run the full quality suite
make lint           # Ruff linter
make typecheck      # Mypy strict mode
make test           # Pytest with coverage

# Train models (after placing Kaggle data in data/raw/)
make train

# Launch the inference API
make serve          # → http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology | Why |
|:------|:-----------|:----|
| Data Processing | **Polars** (LazyFrame/Streaming) | Multithreaded SIMD execution, zero-copy memory, 5-10× faster than Pandas |
| Ad-hoc Analysis | **DuckDB** | In-process OLAP for notebooks and exploratory queries |
| Forecasting | **LightGBM** + **CatBoost** | State-of-the-art gradient boosting for tabular time-series |
| Optimization | **SciPy** + Monte Carlo | Stochastic *(s, S)* policy optimization with spoilage dynamics |
| API Serving | **FastAPI** + **Pydantic v2** | Async, auto-documented (OpenAPI), type-safe validation |
| Package Management | **uv** + `pyproject.toml` | 10-100× faster than pip, deterministic resolution |
| Code Quality | **ruff** + **mypy** (strict) | Unified linter/formatter + static type verification |
| CI/CD | **GitHub Actions** | Lint + typecheck + test on every push and PR |

---

## Feature Engineering Summary

The pipeline generates **30+ features** organized in two modules:

### Deterministic Features ([`temporal.py`](src/features/temporal.py))
*Zero leakage risk — derived only from the calendar date.*

- **Basic calendar**: day_of_week, day_of_month, month, quarter, year, is_weekend, is_month_start/end
- **Cyclical encodings**: sin/cos transforms for circular continuity (Dec ↔ Jan)
- **Ecuadorian economic calendar**: payday_15, payday_end, days_since_payday
- **Holiday proximity**: days_to/from nearest national holiday, bridge-day flags
- **Oil price signals**: lagged WTI price (Ecuador's economy is oil-dependent)

### History-Based Features ([`aggregations.py`](src/features/aggregations.py))
*Leakage-guarded — all lags enforce `≥ 16 day` minimum offset.*

- **Pure lags**: sales at t-16, t-17, t-18, t-21, t-28, t-42, t-56
- **Rolling statistics**: mean, std, median over 7/14/28/56-day windows (shifted ≥16d)
- **Expanding mean**: Cumulative average over all past observations
- **Diff features**: Week-over-week and month-over-month sales changes
- **Promotion lags**: Lagged promotion flags to capture "pantry loading" effects

---

## Author

- **Pauline Lynnea** ([@asdaasp-hearn-mat](https://github.com/asdaasp-hearn-mat)) — [asdaasp@proton.me](mailto:asdaasp@proton.me)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Dataset**: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) by Corporación Favorita via Kaggle.
- **Polars**: [pola-rs/polars](https://github.com/pola-rs/polars) — the high-performance DataFrame library that makes this pipeline fly.
