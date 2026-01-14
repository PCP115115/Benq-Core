# 🧠 Engine: Market Intelligence & Regime Detection

This module constitutes the **Unsupervised Artificial Intelligence** layer of the project (**Context Engine**). Its objective is to infer the latent state or "Market Regime" (e.g., *Bull Market*, *Crash*, *Sideways Trend*) by analyzing the multidimensional structure of technical indicators.

It uses a hybrid architecture of **Deep Learning (PyTorch)** and **Probabilistic Machine Learning (Scikit-Learn)** to transform noisy data into clear and actionable state signals.

---

## 🏗️ System Architecture

The inference pipeline follows a sequential flow designed to maximize robustness and adaptability:

### 1. Dynamic Pre-processing (`Robust Scaler`)
Unlike static features, context requires adaptability.
* **Logic:** Input indicators are normalized "on the fly" using a quarterly sliding window (`NORMALIZATION_WINDOW` in `config.py`).
* **Objective:** Ensure the model understands relative volatility. A VIX of 20 might be high in 2017 but low in 2008. Dynamic normalization corrects this historical bias.

### 2. Latent Feature Extraction (`LSTM Autoencoder`)
A Recurrent Neural Network (LSTM) designed to clean noise.
* **Input:** Temporal sequence ($T=20$) of selected macro indicators (Volatility, Liquidity, Efficiency, Correlation).
* **Compression:** The model attempts to replicate the input by passing it through a "bottleneck".
* **Output (Embeddings):** A compressed vector (10 dimensions) that represents the **essential structure** of the market at that moment, discarding random noise.

### 3. Regime Detection (`GMM Clustering` with Anchoring)
A *Gaussian Mixture Model* classifies the latent vectors.
* **Logic:** Groups days that have similar structural characteristics.
* **Semantic Stability (Semantic Sorting):** After training, the system measures the average volatility of each cluster and automatically reorders them.
* **Output:**
    * `market_regime`: Integer (0 to 4). **Guaranteed**: 0 = Lowest Volatility, 4 = Highest Volatility.
    * `regime_probability`: Statistical probability of belonging to that regime (Model confidence).

---

## 🛡️ "Auto-Healing" System

The module is completely autonomous in managing the AI model lifecycle.

1. **Cold Start:** When invoked, it checks if trained weights exist in `data/models/`.
2. **Automatic Training:** If they do not exist, it automatically downloads the entire available market history, trains the Autoencoder and GMM from scratch, applies semantic sorting, and saves the artifacts.
3. **Inference:** If the models exist, it loads them into memory (CPU/GPU) and executes the prediction.

---

## 🛠️ Configuration and Parameters

The AI behavior is controlled from `src/engine/config.py` under the dictionary `CONTEXT_PARAMS`.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Data Input** | | |
| `INPUT_FEATURES` | `[vol_yz_20d, ...]` | **Important:** The first feature in the list is used as an "Anchor" to sort the regimes (usually Volatility). |
| `NORMALIZATION_WINDOW` | **63** | Normalization window (Trading quarter). |
| **LSTM Architecture** | | |
| `LSTM_WINDOW_SIZE` | **20** | Model short-term memory (1 trading month). |
| `LSTM_HIDDEN_DIM` | **32** | Neurons in the hidden layer. |
| `LSTM_LATENT_DIM` | **10** | Size of the compressed vector. |
| **Clustering** | | |
| `GMM_N_COMPONENTS` | **5** | Number of market regimes to detect. |

---

## 🚀 Usage API: `master_context.py`

The function `get_market_regime` is the single entry point.

### Implementation Example

```python
from src.engine.context import master_context

# Obtener el régimen de mercado
df_regime = master_context.get_market_regime(tickers="AAPL")

# Interpretación GARANTIZADA gracias al Anclaje Semántico:
# Régimen 0: Calma / Baja Volatilidad (Ideal para Carry Trade / Tendencia)
# Régimen 4: Pánico / Alta Volatilidad (Crash o Rebote violento)
current_regime = df_regime.tail(1)["market_regime"].item()

if current_regime >= 3:
    print("⚠️ Mercado en régimen de Alta Volatilidad. Reducir exposición.")
elif current_regime == 0:
    print("✅ Mercado en calma. Condiciones estables.")