# Mini-Models Module: Mixture of Experts (MoE) Architecture

**Version:** 1.0.0  
**Status:** Production Ready  
**Module Path:** `src/engine/mini_models/`  
**Author:** Pedro (Lead Architect)

---

## 1. Executive Summary

The **Mini-Models Module** constitutes the feature extraction and probabilistic inference engine of the Benq-Core trading system. Instead of relying on a single monolithic model to predict price movements directly from raw data, this architecture implements a **Mixture of Experts (MoE)** approach.

Three independent, specialized "Expert" models analyze the market simultaneously for each asset. They do not output trading signals (Buy/Sell); rather, they output **probabilities (0 to 1)** regarding specific market phenomena. These probabilities serve as high-level features (Meta-Features) for the downstream Meta-Model (The Strategy Brain).

---

## 2. Architecture & Components

The system is designed with a **Shared-Nothing Architecture** regarding the models, but centralized orchestration for data loading to maximize computational efficiency.

### 2.1. The Three Experts
For every ticker in the universe, the system trains and queries three distinct GBM (Gradient Boosting Machine) classifiers:

#### A. Trend Expert (`trend_mini_model.py`)
* **Objective:** Detect directional momentum and sustained moves.
* **Key Features:** ADX (Trend Strength), MACD, Relative SMAs, Kaufman Efficiency Ratio.
* **Mathematical Question:** *Is the price likely to hit a dynamic Upper Barrier before a Lower Barrier (or vice versa) within the forecast horizon?*
* **Outputs:**
    * `P_Trend_Up`: Probability of a bullish trend.
    * `P_Trend_Down`: Probability of a bearish trend.

#### B. Reversion Expert (`reversion_mini_models.py`)
* **Objective:** Detect overextended conditions and mean reversion potential (Counter-trend).
* **Key Features:** RSI, Volatility Bands (Z-Scores), Skewness, Divergences.
* **Mathematical Question:** *Is the price at a statistical extreme relative to its volatility context?*
* **Outputs:**
    * `P_Rev_Up`: Probability of a bounce (Long Reversion).
    * `P_Rev_Down`: Probability of a pullback (Short Reversion).

#### C. Volatility Expert (`volatility_mini_models.py`)
* **Objective:** Detect Market Regime changes (Risk Management).
* **Key Features:** Yang-Zhang, Garman-Klass, Parkinson Estimators, Amihud Illiquidity.
* **Mathematical Question:** *Is the realized volatility of the future horizon likely to exceed the current implied volatility?*
* **Outputs:**
    * `P_Vol_Exp`: Probability of Volatility Expansion (Risk ON / Storm).
    * `P_Vol_Comp`: Probability of Volatility Compression (Risk OFF / Calm).

### 2.2. The Orchestrator (`master_mini_models.py`)
The Master script handles the lifecycle of the training and inference process:
* **Multiprocessing Strategy:** Utilizes all available CPU cores. Each core processes a distinct Ticker to minimize memory overhead ("Parallel Tickers, Sequential Models").
* **Optimization:** Inside each core, the three experts are trained sequentially using highly optimized LightGBM threads (`n_jobs=1`), ensuring 100% CPU utilization without thread contention.
* **Pipeline:** `Load Data` -> `Train Experts` -> `Inference on Test Set` -> `Save Output`.

---

## 3. Engineering & Safety Mechanisms

To ensure the statistical validity of the models and prevent "paper profits" that fail in live trading, the following mechanisms are strictly enforced:

### 3.1. Purged Time-Series Split
We strictly avoid standard random shuffling (`train_test_split`). Instead, we implement a **Purged Split**:
1.  **Chronological Order:** Training data always precedes Test data.
2.  **Safety Gap (Purge):** A specific number of observations (equal to the `FORECAST_HORIZON`) are deleted between the Training set and the Test set.
3.  **Purpose:** This eliminates **Look-Ahead Bias** and **Data Leakage**, ensuring that the model training phase never "sees" labels that overlap with the testing period.

### 3.2. Robust Data Handling
* **Raw vs. Robust Layers:** The system loads a complete data context allowing for on-the-fly target calculation using Raw prices (High/Low) while training on normalized/robust features to ensure stationarity.
* **Auto-Healing:** The pipeline automatically handles missing models or first-time runs, skipping tickers with insufficient data without crashing the main process.

---

## 4. Integration Guide: The Meta-Model Interface

The Mini-Models module produces a standardized output that serves as the **Input Layer** for the Meta-Model.

### 4.1. Output Format
For each processed ticker (e.g., AAPL), the Master generates a Parquet file at:
`src/data/processed/meta_model_inputs/meta_input_AAPL.parquet`

**Table Structure (The "Expert Opinions"):**

| Date | Ticker | Close | Log_Ret | P_Trend_Up | P_Trend_Down | P_Rev_Up | P_Rev_Down | P_Vol_Exp | P_Vol_Comp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ... | AAPL | 150.5 | 0.012 | **0.85** | 0.12 | 0.05 | 0.60 | **0.92** | 0.08 |

### 4.2. Implementation in Meta-Model (Future Work)
To utilize this module in the future `meta_model.py`, call the orchestration function directly. This function abstracts all complexity (training, inference, file saving).

#### Python Implementation Example

```python
# src/engine/meta_model/meta_model_main.py

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- IMPORT THE MINI-MODELS ORCHESTRATOR ---
from engine.mini_models.src_mini_models import master_mini_models

def run_meta_model_workflow(tickers: list):
    """
    Main workflow to train the Meta-Model.
    """
    print("STEP 1: Generating Expert Opinions (Mini-Models)...")
    
    # ---------------------------------------------------------
    # CALLING THE MODULE
    # This single line trains all 3 experts for all tickers
    # and generates the input parquet files.
    # ---------------------------------------------------------
    master_mini_models.run_mini_models_pipeline(tickers=tickers, n_jobs=-1)
    
    print("STEP 2: Loading Expert Probabilities...")
    
    # Example: Loading the generated data for AAPL
    import polars as pl
    input_path = project_root / "src/data/processed/meta_model_inputs/meta_input_AAPL.parquet"
    
    if input_path.exists():
        df_meta_input = pl.read_parquet(input_path)
        print(f"Loaded {df_meta_input.height} rows of probabilities for AAPL.")
        print(df_meta_input.head())
        
        # ... Next steps: Train the Meta-Model (Reinforcement Learning or Supervised) ...
    else:
        print("Error: Input files not generated.")

if __name__ == "__main__":
    universe = ["AAPL", "MSFT", "TSLA", "BAC"]
    run_meta_model_workflow(universe)

