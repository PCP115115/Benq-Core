# ⚙️ Engine: Feature Engineering & Alpha Generation

Este módulo constituye el núcleo de investigación cuantitativa del proyecto (**Alpha Engine**). Su objetivo es transformar series de precios OHLCV "sucias" y no estacionarias en una matriz de factores limpia, normalizada y neutralizada, lista para alimentar modelos de Machine Learning.

Utiliza **Polars** para una ejecución vectorizada, perezosa (*Lazy*) y multihilo, permitiendo calcular cientos de indicadores sobre miles de activos en cuestión de segundos. El sistema cubre activos de **Norteamérica, Europa, Asia-Pacífico y Latam**.

---

## 🧠 Filosofía de Diseño: El Sistema de 3 Capas

Para maximizar la relación señal-ruido (*Signal-to-Noise Ratio*), los datos atraviesan tres etapas de refinamiento secuencial. El usuario puede extraer los datos en cualquiera de estas fases mediante el parámetro `layer`.

### 1. Raw Layer (Capa Cruda)
Cálculo puramente matemático de indicadores técnicos avanzados y métricas de microestructura de mercado.
* **Fuente:** `src_features/indicators.py`
* **Indicadores Clave:**
    * **Momentum:** RSI (con *Wilder's Smoothing*), MACD.
    * **Volatilidad (Estimadores Institucionales):** * *Yang-Zhang:* [NUEVO] El estimador más robusto. Captura saltos de apertura (*Overnight Gaps*) y es independiente de la tendencia (*Drift*).
        * *Garman-Klass:* Volatilidad eficiente basada en OHLC.
        * *Parkinson:* Volatilidad basada en rangos (High-Low).
        * *Histórica:* Desviación estándar de retornos logarítmicos.
    * **Estructura de Mercado:**
        * *Volatility Cones:* [NUEVO] Precios teóricos de Techo y Suelo proyectados a futuro ($T+5$) basados en la volatilidad Yang-Zhang. Actúan como soportes y resistencias dinámicos.
    * **Liquidez:** Ratio de Iliquidez de Amihud (Impacto del volumen en el precio).
    * **Eficiencia:** Kaufman Efficiency Ratio (KER).
* **Problema:** Estos valores tienen escalas incomparables (ej. RSI $\in [0,100]$, Amihud $\approx 10^{-6}$, Precio $\in [10, 1000]$).

### 2. Robust Layer (Normalización Temporal)
Resuelve el problema de la **estacionariedad**. Cada indicador se normaliza respecto a su propia historia reciente (ventana deslizante) utilizando un escalado robusto a valores atípicos (*outliers*).

$$Z_{rob} = \frac{X_t - \text{Median}(X_{t-n...t})}{\text{IQR}(X_{t-n...t})}$$

* **Lógica:** Compara el valor de hoy contra los últimos $N$ días (ej. 1 año bursátil).
* **Resultado:** Una serie centrada en 0, donde valores $>2$ o $<-2$ representan anomalías estadísticas reales del activo.
* **Sufijo:** `_rob` (ej. `vol_yz_20d_rob`).

### 3. Neutral Layer (Neutralización Sectorial)
Resuelve el problema de la **correlación de mercado (Beta)**. Aísla el rendimiento idiosincrático (Alpha) del activo eliminando la tendencia del sector en ese día específico.

* **Lógica:** Agrupa todos los activos de un mismo sector en la fecha $T$, calcula la mediana del sector y reajusta el score del activo.
* **Resultado:** Un valor que indica qué tan bueno es el activo *comparado con sus pares* hoy. Ideal para estrategias *Long/Short* y *Market Neutral*.
* **Sufijo:** `_neutral` (ej. `vol_yz_20d_neutral`).

---

## 🛠️ Configuración y Parámetros

No es necesario editar el código fuente para ajustar los hiperparámetros. Todo el comportamiento se controla desde `config.py`.

### Parámetros de Indicadores (`FEATURES_PARAMS`)

| Parámetro | Valor Defecto | Descripción |
| :--- | :--- | :--- |
| **Volatilidad** | | |
| `YANG_ZHANG_WINDOW` | **20** | [NUEVO] Ventana para el estimador Yang-Zhang (Drift+Gaps). |
| `GARMAN_KLASS_WINDOW`| **20** | Ventana para volatilidad eficiente (OHLC). |
| `PARKINSON_WINDOW` | **20** | Ventana para volatilidad de rango (High-Low). |
| `VOLATILITY_WINDOW` | **20** | Ventana para volatilidad histórica (Close-Close). |
| **Conos de Volatilidad** | | |
| `YZ_Z_SCORE` | **1.96** | [NUEVO] Intervalo de confianza (95%) para los conos. |
| `YZ_FORECAST_HORIZON` | **5** | [NUEVO] Días de proyección para el cálculo de Techo/Suelo. |
| **Momentum/Otros** | | |
| `RSI_PERIOD` | **14** | Periodo para el oscilador RSI (Wilder). |
| `AMIHUD_WINDOW` | **20** | Ventana suavizado para ratio de iliquidez. |
| `AMIHUD_SCALING` | **1e6** | Factor multiplicador para hacer legible el ratio Amihud. |
| `SKEW_WINDOW` | **60** | Ventana para calcular asimetría (Skewness) trimestral. |
| `CORR_WINDOW` | **20** | Ventana para correlación Precio-Volumen. |
| `KER_WINDOW` | **10** | Ventana para Eficiencia de Kaufman. |

### Parámetros de Normalización (`NORMALIZATION_PARAMS`)

| Parámetro | Valor Defecto | Descripción |
| :--- | :--- | :--- |
| `ROLLING_WINDOW` | **252** | Historia para calcular Z-Score (1 año bursátil). |
| `MIN_PERIODS` | **120** | Mínimo de datos requeridos al inicio para evitar ruido. |
| `MIN_ASSETS_PER_SECTOR`| **5** | Mínimo de activos para aplicar neutralización sectorial. |

---

## 🚀 API de Uso: `master_features.py`

La función `get_feature_matrix` es el punto de entrada único (*Single Entry Point*). 

Cuenta con un sistema de **"Auto-Healing"**: si los datos procesados no existen o están corruptos, invoca automáticamente al pipeline de cálculo (`pipeline_features.py`) para regenerarlos antes de devolver el resultado.

### Firma de la Función

```python
def get_feature_matrix(
    tickers: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    layer: str = "all",
    features: Optional[Union[str, List[str]]] = None
) -> pl.DataFrame