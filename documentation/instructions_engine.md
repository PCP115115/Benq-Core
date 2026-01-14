# ⚙️ Benq-Core: Engine Workflow Overview (v1.0.0)

This document outlines the sequential data flow of the **Benq-Core Engine**. The architecture is designed as a **Feature-Driven, Hierarchical Stack**, where each layer refines the data for the next one.

---

## 🗺️ The 4-Layer Architecture

The system transforms raw market data into actionable probabilities through four distinct stages:

### 1. The Alpha Layer (Features)
**Input:** Raw OHLCV (Open, High, Low, Close, Volume).
**Process:**
* Calculates technical indicators (RSI, MACD, Bollinger, etc.).
* Calculates advanced microstructure metrics (Yang-Zhang Volatility, Amihud Liquidity, Kaufman Efficiency).
* **Normalization:** Applies Robust Scaling (Rolling Z-Score) to make data stationary.
* **Neutralization:** (Optional) Removes sector beta.
**Output:** A clean Feature Matrix ($X$).
**Code:** `src/engine/src_features/master_features.py`

### 2. The Intelligence Layer (Context)
**Input:** Selected macro-features from Layer 1.
**Process:**
* **LSTM Autoencoder:** Compresses market noise into a latent vector representation.
* **GMM (Gaussian Mixture Model):** Clusters these vectors into discrete Market Regimes (0=Calm ... 2=Crisis).
**Output:** `market_regime` (Integer) and `regime_probability`.
**Code:** `src/engine/context/master_context.py`

### 3. The Expert Layer (Mini-Models)
**Input:** Feature Matrix + Specific specialized indicators.
**Process:**
* **Trend Expert:** Predicts directional continuations.
* **Reversion Expert:** Predicts price reversions.
* **Volatility Expert:** Predicts Volatility Expansion/Compression regimes.
**Output:** A set of probabilities ($P_{trend}$, $P_{rev}$, $P_{vol}$).
**Code:** `src/engine/mini_models/src_mini_models/master_mini_models.py`

### 4. The Decision Layer (Meta-Model)
**Input:** All outputs from Layers 1, 2, and 3.
**Process:**
* **Stacking:** Uses XGBoost to learn which Expert is reliable in the current Market Regime.
* **Dual Target:** Predicts the probability of the price hitting a dynamic upper or lower volatility barrier.
**Output:** Final Signals (`Prob_UP`, `Prob_DOWN`).
**Code:** `src/engine/meta_model/src_meta_model/pipeline_meta.py`

---

## 🔄 Execution Pipeline

To generate a signal for a specific asset (e.g., AAPL), the system follows this strict dependency chain:

1.  **`get_feature_matrix("AAPL")`**: Checks for cached data. If missing, downloads and computes Layer 1.
2.  **`get_market_regime("AAPL")`**: Checks for trained GMM models. If missing, retrains context models and classifies the current day.
3.  **`run_mini_models_pipeline(["AAPL"])`**: Trains/Loads the 3 experts and generates their probability predictions (saved as intermediate Parquet files).
4.  **`train_meta_model("AAPL")`**: Aggregates all previous data, calculates the "Race Logic" targets, and trains the final XGBoost classifiers.

### Production Logic
In production (Live Trading), the signals outputed by the meta-model will be optimized and proccesed by the next block: strategy, where the probabilities of the meta-model will be traduced into a mathematically optimized investment strategy.