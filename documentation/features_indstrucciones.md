## Documentación del Motor de Features (Feature Engine)

**Nombre del archivo:** `src/engine/README.md`

Esta versión se enfoca en la metodología cuantitativa: por qué se hacen las cosas así (estacionariedad, neutralización) y cómo esto ayuda a los modelos de ML.


# ⚙️ Feature Engineering & Alpha Generation

El módulo `engine` es el núcleo de investigación cuantitativa del proyecto. Su objetivo es transformar series de precios "sucias" y no estacionarias en una matriz de factores (**Alpha Factors**) limpia, normalizada y lista para alimentar modelos de Machine Learning.

Utiliza **Polars** para una ejecución vectorizada y *Lazy*, permitiendo calcular cientos de indicadores sobre miles de activos en segundos.

## 🧠 Filosofía de Diseño: El Sistema de 3 Capas

Para maximizar la relación señal-ruido, los datos pasan por tres etapas de refinamiento, accesibles a través de la API.

### 1. Raw Layer (Capa Cruda)
Cálculo puramente matemático de indicadores técnicos sobre la serie de precios.
* *Fuente:* `src/engine/src_features/indicators.py`
* *Ejemplos:* RSI de 14 días, Bandas de Bollinger, Volatilidad de Parkinson.
* *Problema:* Estos valores tienen escalas muy diferentes (RSI 0-100, Precio 10-1000) y no son comparables entre activos.

### 2. Robust Layer (Normalización Temporal)
Resuelve el problema de la **estacionariedad**. Cada indicador se normaliza respecto a su propia historia reciente (Ventana rodante, ej. 252 días) usando un escalado robusto a outliers.
* *Fórmula:* $Z_{rob} = \frac{X_t - Median(X_{t-n...t})}{IQR(X_{t-n...t})}$
* *Resultado:* Una serie centrada en 0 donde los valores extremos representan anomalías estadísticas reales del activo.
* *Sufijo:* `_rob` (ej. `rsi_14_rob`).

### 3. Neutral Layer (Neutralización Sectorial)
Resuelve el problema de la **correlación de mercado (Beta)**. Aísla el rendimiento idiosincrático del activo eliminando la tendencia del sector en ese día específico.
* *Lógica:* Agrupa todos los activos de un sector en una fecha $T$, calcula la mediana del sector y reajusta el score del activo.
* *Resultado:* Un valor que indica qué tan bueno es el activo *comparado con sus pares* hoy. Ideal para estrategias *Long/Short*.
* *Sufijo:* `_neutral` (ej. `rsi_14_neutral`).

---
## Parámetros existentes:
- **layer**: Define qué columnas devolver: "raw" (indicadores puros), "robust" (z-score temporal), "neutral" (z-score sectorial) o "all" (todo)
## 🚀 API Principal: `master_features.py`
- **tickers**: Filtra que activos cargar.
- **start_date**: Fecha de inicio
- **end_date**: fecha de final

La función `get_feature_matrix` orquesta la carga, cálculo (si es necesario) y filtrado de la matriz de características.

```python
from src.engine.src_features.master_features import get_feature_matrix

# Obtener la matriz lista para ML (Capa Neutral)
# Filtra automáticamente columnas meta e indicadores intermedios.
df_ml_ready = get_feature_matrix(
    start_date="2022-01-01", 
    layer="neutral"
)

# Obtener datos para depuración (Capa Raw + Robust)
df_debug = get_feature_matrix(
    tickers=["AAPL", "MSFT"],
    layer="robust"
)


