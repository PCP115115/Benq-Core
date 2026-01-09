# 🧠 Engine: Market Intelligence & Regime Detection

Este módulo constituye la capa de **Inteligencia Artificial No Supervisada** del proyecto (**Context Engine**). Su objetivo es inferir el estado latente o "Régimen de Mercado" (ej. *Bull Market*, *Crash*, *Lateralidad*) analizando la estructura multidimensional de los indicadores técnicos.

Utiliza una arquitectura híbrida de **Deep Learning (PyTorch)** y **Machine Learning Probabilístico (Scikit-Learn)** para transformar datos ruidosos en señales de estado claras y accionables.

---

## 🏗️ Arquitectura del Sistema

El pipeline de inferencia sigue un flujo secuencial diseñado para maximizar la robustez y adaptabilidad:

### 1. Pre-procesamiento Dinámico (`Robust Scaler`)
A diferencia de los features estáticos, el contexto requiere adaptabilidad.
* **Lógica:** Se normalizan los indicadores de entrada "al vuelo" utilizando una ventana deslizante trimestral (`NORMALIZATION_WINDOW` en `config.py`).
* **Objetivo:** Asegurar que el modelo entienda la volatilidad relativa. Un VIX de 20 puede ser alto en 2017 pero bajo en 2008. La normalización dinámica corrige este sesgo histórico.

### 2. Extracción de Features Latentes (`LSTM Autoencoder`)
Una red neuronal recurrente (LSTM) diseñada para limpiar ruido.
* **Input:** Secuencia temporal ($T=20$) de indicadores macro seleccionados (Volatilidad, Liquidez, Eficiencia, Correlación).
* **Compresión:** El modelo intenta replicar la entrada pasándola por un "cuello de botella" (*Bottleneck*).
* **Output (Embeddings):** Un vector comprimido (10 dimensiones) que representa la **estructura esencial** del mercado en ese momento, descartando el ruido aleatorio.

### 3. Detección de Régimen (`GMM Clustering` con Anclaje)
Un modelo de Mezcla Gaussiana (*Gaussian Mixture Model*) clasifica los vectores latentes.
* **Lógica:** Agrupa los días que tienen características estructurales similares.
* **Estabilidad Semántica (Semantic Sorting):** Tras el entrenamiento, el sistema mide la volatilidad promedio de cada clúster y los reordena automáticamente.
* **Salida:**
    * `market_regime`: Entero (0 a 4). **Garantizado**: 0 = Menor Volatilidad, 4 = Mayor Volatilidad.
    * `regime_probability`: Probabilidad estadística de pertenecer a ese régimen (Confianza del modelo).

---

## 🛡️ Sistema "Auto-Healing"

El módulo es completamente autónomo en su gestión del ciclo de vida de los modelos IA.

1. **Cold Start:** Al invocarse, verifica si existen los pesos entrenados en `data/models/`.
2. **Entrenamiento Automático:** Si no existen, descarga automáticamente todo el histórico de mercado disponible, entrena el Autoencoder y el GMM desde cero, aplica el ordenamiento semántico y guarda los artefactos.
3. **Inferencia:** Si los modelos existen, los carga en memoria (CPU/GPU) y ejecuta la predicción.

---

## 🛠️ Configuración y Parámetros

El comportamiento de la IA se controla desde `src/engine/config.py` bajo el diccionario `CONTEXT_PARAMS`.

| Parámetro | Valor Defecto | Descripción |
| :--- | :--- | :--- |
| **Entrada de Datos** | | |
| `INPUT_FEATURES` | `[vol_yz_20d, ...]` | **Importante:** La primera feature de la lista se usa como "Ancla" para ordenar los regímenes (usualmente Volatilidad). |
| `NORMALIZATION_WINDOW` | **63** | Ventana de normalización (Trimestre bursátil). |
| **Arquitectura LSTM** | | |
| `LSTM_WINDOW_SIZE` | **20** | Memoria corta del modelo (1 mes bursátil). |
| `LSTM_HIDDEN_DIM` | **32** | Neuronas en la capa oculta. |
| `LSTM_LATENT_DIM` | **10** | Tamaño del vector comprimido. |
| **Clustering** | | |
| `GMM_N_COMPONENTS` | **5** | Número de regímenes de mercado a detectar. |

---

## 🚀 API de Uso: `master_context.py`

La función `get_market_regime` es el punto de entrada único (*Single Entry Point*).

### Ejemplo de Implementación

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