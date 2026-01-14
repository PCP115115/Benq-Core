# 🧠 Engine: Meta-Model & Decision Core

This module constitutes the **Supervised Learning & Stacking** layer of the project (**Benq-core**). It acts as the "Director of Operations," aggregating information from all previous layers (Technical Features, Market Regime, and Expert Opinions) to make a final probabilistic decision.

It uses **XGBoost (Extreme Gradient Boosting)** to learn the non-linear relationships between the experts' success rates and the current market context.

---

## 🏗️ System Architecture

The Meta-Model is designed as an **Adaptative Mixture of Experts (AMoE)**. It does not look at the price directly to predict; instead, it looks at *who is right* under the current conditions.

### 1. Data Aggregation (`download_meta.py`)
Before training, the system builds a "Super-Dataset" by performing an inner join of four data sources:
1.  **Sectorially Normalized Features:** (RSI, MACD, etc.) from `master_features.py`.
2.  **Raw Data:** Real prices and raw volatility (needed for physical barrier calculation).
3.  **Market Context:** The Regime (0-4) and its probability from `master_context.py`. The amount of regime can be changed in config.py.
4.  **Expert Probabilities:** The predictions ($P \in [0,1]$) from Trend, Reversion, and Volatility mini-models.

### 2. The Target Logic: "Race Logic" (Dual Breakout)
The model does not predict fixed direction. It predicts a **Volatility Breakout Event**.
It uses a **Triple Barrier Method** with dynamic volatility bounds:

$$\text{Upper Barrier} = \text{Close}_t + (\text{Vol}_{YZ} \times Z_{score} \times \sqrt{T})$$
$$\text{Lower Barrier} = \text{Close}_t - (\text{Vol}_{YZ} \times Z_{score} \times \sqrt{T})$$

* **TARGET_UP = 1:** If the price hits the **Upper Barrier** *before* hitting the Lower Barrier within the forecast horizon ($T$).
* **TARGET_DOWN = 1:** If the price hits the **Lower Barrier** *before* hitting the Upper Barrier.
* **0:** If price stays within the channel (noise/sideways).

### 3. The Learner (`XGBoost`)
Two separate classifiers are trained:
* **`xgboost_meta_up.json`**: Specialized in detecting Long opportunities.
* **`xgboost_meta_down.json`**: Specialized in detecting Short/Crash opportunities.

**Key Features:**
* **Dynamic Class Balancing:** Automatically calculates `scale_pos_weight` to handle rare events (e.g., market crashes).
* **Purged Time-Series Split:** Prevents data leakage by enforcing a "gap" between training and testing data equal to the forecast horizon.

---

## 🛠️ Configuration and Parameters

Behavior is controlled via `src/engine/config.py` under `META_MODEL_PARAMS`.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **Target Definition** | | |
| `FORECAST_HORIZON` | **5** | Days to look ahead for the breakout race. |
| `YZ_Z_SCORE` | **1.96** | Volatility multiplier (Standard Deviations) for barriers. |
| **XGBoost Hyperparams** | | |
| `max_depth` | **4** | Tree depth. Low value prevents overfitting. |
| `learning_rate` | **0.02** | Step size shrinkage. Low value requires more trees but generalizes better. |
| `n_estimators` | **500** | Number of boosting rounds. |
| `colsample_bytree` | **0.6** | Fraction of features (Experts) used per tree. Forces model diversity. |

---

## 🚀 Usage API: `pipeline_meta.py`

The script `src/engine/meta_model/src_meta_model/pipeline_meta.py` manages the training lifecycle.

### How to Train the Model

To retrain the model with the latest available data (configured in `config.py`):

```python
from src.engine.meta_model.src_meta_model import pipeline_meta

# This function triggers the full flow:
# Download -> Aggregate -> Calculate Targets -> Train XGBoost -> Save JSON
pipeline_meta.train_meta_model(ticker="AAPL")