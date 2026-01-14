***

### `features_instructions.md`


# ⚙️ Engine: Feature Engineering & Alpha Generation

This module constitutes the quantitative research core of the project (**Alpha Engine**). Its objective is to transform "dirty" and non-stationary OHLCV price series into a clean, normalized, and neutralized factor matrix, ready to feed Machine Learning models.

It uses **Polars** for vectorized, lazy, and multi-threaded execution, allowing hundreds of indicators to be calculated on thousands of assets in a matter of seconds. The system covers assets from **North America, Europe, Asia-Pacific, and Latam**.

---

## 🧠 Design Philosophy: The 3-Layer System

To maximize the *Signal-to-Noise Ratio*, data goes through three stages of sequential refinement. The user can extract data at any of these phases using the `layer` parameter.

### 1. Raw Layer
Purely mathematical calculation of advanced technical indicators and market microstructure metrics.
* **Source:** `src_features/indicators.py`
* **Key Indicators:**
    * **Momentum:** RSI (with *Wilder's Smoothing*), MACD.
    * **Volatility (Institutional Estimators):**
        * *Yang-Zhang:* [NEW] The most robust estimator. Captures opening jumps (*Overnight Gaps*) and is independent of the trend (*Drift*).
        * *Garman-Klass:* Efficient volatility based on OHLC.
        * *Parkinson:* Range-based volatility (High-Low).
        * *Historical:* Standard deviation of logarithmic returns.
    * **Market Structure:**
        * *Volatility Cones:* [NEW] Theoretical Ceiling and Floor prices projected into the future ($T+5$) based on Yang-Zhang volatility. Act as dynamic support and resistance.
    * **Liquidity:** Amihud Illiquidity Ratio (Impact of volume on price).
    * **Efficiency:** Kaufman Efficiency Ratio (KER).
* **Problem:** These values have incomparable scales (e.g., RSI $\in [0,100]$, Amihud $\approx 10^{-6}$, Price $\in [10, 1000]$).

### 2. Robust Layer (Temporal Normalization)
Solves the **stationarity** problem. Each indicator is normalized relative to its own recent history (sliding window) using robust scaling to outliers.

$$Z_{rob} = \frac{X_t - \text{Median}(X_{t-n...t})}{\text{IQR}(X_{t-n...t})}$$

* **Logic:** Compares today's value against the last $N$ days (e.g., 1 trading year).
* **Result:** A series centered at 0, where values $>2$ or $<-2$ represent real statistical anomalies of the asset.
* **Suffix:** `_rob` (e.g., `vol_yz_20d_rob`).

### 3. Neutral Layer (Sector Neutralization)
Solves the **market correlation (Beta)** problem. Isolates the asset's idiosyncratic performance (Alpha) by removing the sector trend on that specific day.

* **Logic:** Groups all assets of the same sector on date $T$, calculates the sector median, and readjusts the asset score.
* **Result:** A value indicating how good the asset is *compared to its peers* today. Ideal for *Long/Short* and *Market Neutral* strategies.
* **Suffix:** `_neutral` (e.g., `vol_yz_20d_neutral`).

---

## 🛠️ Configuration and Parameters

It is not necessary to edit the source code to adjust hyperparameters. All behavior is controlled from `config.py`.

### Indicator Parameters (`FEATURES_PARAMS`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Volatility** | | |
| `YANG_ZHANG_WINDOW` | **20** | [NEW] Window for Yang-Zhang estimator (Drift+Gaps). |
| `GARMAN_KLASS_WINDOW`| **20** | Window for efficient volatility (OHLC). |
| `PARKINSON_WINDOW` | **20** | Window for range volatility (High-Low). |
| `VOLATILITY_WINDOW` | **20** | Window for historical volatility (Close-Close). |
| **Volatility Cones** | | |
| `YZ_Z_SCORE` | **1.96** | [NEW] Confidence interval (95%) for cones. |
| `YZ_FORECAST_HORIZON` | **5** | [NEW] Projection days for Ceiling/Floor calculation. |
| **Momentum/Others** | | |
| `RSI_PERIOD` | **14** | Period for RSI oscillator (Wilder). |
| `AMIHUD_WINDOW` | **20** | Smoothing window for illiquidity ratio. |
| `AMIHUD_SCALING` | **1e6** | Multiplier factor to make Amihud ratio readable. |
| `SKEW_WINDOW` | **60** | Window to calculate quarterly Skewness. |
| `CORR_WINDOW` | **20** | Window for Price-Volume correlation. |
| `KER_WINDOW` | **10** | Window for Kaufman Efficiency. |

### Normalization Parameters (`NORMALIZATION_PARAMS`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `ROLLING_WINDOW` | **252** | History to calculate Z-Score (1 trading year). |
| `MIN_PERIODS` | **120** | Minimum data required at start to avoid noise. |
| `MIN_ASSETS_PER_SECTOR`| **5** | Minimum assets to apply sector neutralization. |

---

## 🚀 Usage API: `master_features.py`

The function `get_feature_matrix` is the single entry point.

It features an **"Auto-Healing"** system: if processed data does not exist or is corrupt, it automatically invokes the calculation pipeline (`pipeline_features.py`) to regenerate it before returning the result.

### Function Signature

```python
def get_feature_matrix(
    tickers: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    layer: str = "all",
    features: Optional[Union[str, List[str]]] = None
) -> pl.DataFrame