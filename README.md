# 🏛️ BenQ-Core

![Status](https://img.shields.io/badge/Status-Active_Development-green?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Modular_Monolith-blue?style=flat-square)
![Licence](https://img.shields.io/badge/license-MIT-purple?style=flat-square)
![Version](https://img.shields.io/badge/version-1_beta-orange?style=flat-square)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-Fast_Dataframe-CD7F32?style=flat-square&logo=polars)
![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

> **Ben Quant Core** is an end-to-end quantitative research framework designed for high-performance stock analysis. It combines robust data engineering with unsupervised deep learning to identify market regimes, model asset behavior, and develop data-driven strategies aimed at achieving consistent alpha.
## 📂 System Architecture

The codebase follows a modular architecture designed for scalability, separating concerns between data ingestion, alpha generation (modeling), and strategy execution.

```text
.
├── documentation/       # Technical manuals and module instructions
└── src/
    ├── src_DD/          # Data Ingestion & Safety Layer (ETL)
    ├── engine/          # Core Analytics & Prediction Engine
    │   ├── src_features/    # Factor Engineering (Raw/Robust/Neutral)
    │   └── context/         # Market Regime Detection (Unsupervised Learning)
    ├── strategy/        # Strategy Development & Backtesting Framework
    └── dashboard/       # Front-end & Visualization (In Development)
```

## 🏗️ Steps of Benq-V1

At the moment: Phase 2

- [x] **Phase 1: Data Infrastructure** 
  - Fault-tolerant ETL pipeline using `yfinance` with exponential backoff.
  - Data quality and integrity control (split detection, gap filling, etc.) via `safety.py`.
  - OLAP storage layer based on **DuckDB** and **Parquet**.
  - Culminates in the function: `MarketLoader()`.

- [x ] **Phase 2: Modeling System Development** 
  - **Features**: Module dedicated to feature engineering, culminating in the function: `get_feature_matrix()`.
  - **Context**: Advanced module for Market Regime Detection.
  - **Mini-models**: Development of specialized models following a Mixture of Experts (MoE) architecture.
  - **Meta-Model**: Integration layer that processes inputs from this phase to generate a global prediction.
  - **Safety Engine**: Security module to verify data integrity and calculation accuracy, including automated error logging.

- [ ] **Phase 3: Strategy & Backtesting** - Development of optimized trading strategies based on the Meta-Model's output, supported by a rigorous and high-performance backtesting system.

- [ ] **Phase 4: Dashboard & Production** - Development of a production-ready front-end with a professional visual interface.
  - Integration of a dedicated module to connect the strategies defined in Phase 3 with major Broker APIs for live execution.
---

<h2 align="center">Detailed documentation can be found in BENQ-CORE/documentation</h2>

## 🚀 Fast start

### Requirements
```bash
pip install -r requirements.txt
```

