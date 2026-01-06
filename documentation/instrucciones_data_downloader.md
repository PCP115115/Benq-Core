# 🏛️ Data Core: Pipeline de Ingesta y Gestión de Datos de Mercado

Este módulo implementa una arquitectura **ETL (Extract, Transform, Load)** de grado institucional diseñada para construir y mantener un *Data Lake* local de series temporales financieras.

A diferencia de scripts de descarga simples, este núcleo prioriza la **integridad de los datos** (detección de splits, auditoría de gaps) y la **eficiencia de consulta** (OLAP via DuckDB).

## 🏗️ Arquitectura del Sistema

El flujo de datos se orquesta a través de tres etapas secuenciales gestionadas por `master_data_loader.py`:

### 1. Extracción Resiliente (`download.py`)
* **Motor:** `yfinance` con *wrappers* personalizados.
* **Concurrencia:** Ejecución multihilo (`ThreadPoolExecutor`) para maximizar el ancho de banda.
* **Lógica Incremental:** Detecta automáticamente la última fecha disponible en disco y descarga solo el *delta* necesario, minimizando el tráfico de red y el tiempo de ejecución.
* **Manejo de Errores:** Implementa *Exponential Backoff* para reintentar peticiones fallidas sin saturar la API del proveedor.

### 2. Transformación y Auditoría (`safety.py`)
Actúa como un *firewall* de calidad antes de que los datos sean consumidos:
* **Alineación de Calendarios:** Cruza cada activo contra un índice de referencia (ej. `^GSPC`, `^IBEX`) para validar días de negociación reales.
* **Detección de Splits:** Algoritmo heurístico que identifica caídas de precio >30% no explicadas por el mercado (caída del benchmark >-25%). Si se detecta, purga el archivo corrupto para forzar una re-descarga limpia.
* **Reparación de Series:** Rellena huecos (*gaps*) menores mediante *Forward Fill* y marca estos registros con `data_quality=0` para que los modelos puedan discriminarlos.

### 3. Carga y Acceso OLAP (`loader.py`)
* **Virtualización:** No carga todos los CSV/Parquet en RAM. Utiliza **DuckDB** para montar una vista SQL virtual sobre los archivos físicos.
* **Zero-Copy:** Transfiere los resultados de las consultas a **Polars** utilizando Apache Arrow, garantizando una latencia mínima incluso con millones de filas.

---

## 🔧 Guía de Uso: `MarketLoader`

La clase `MarketLoader` es la interfaz única de acceso (Facade) al Data Lake.

### Inicialización

```python
from src.src_DD.loader import MarketLoader

# Inicializa el cargador.
# actualizar_datos=True: Verifica la antigüedad de los datos. Si > 24h, ejecuta el pipeline ETL.
# actualizar_datos=False: Modo "Solo Lectura" (más rápido).
loader = MarketLoader(actualizar_datos=False)
