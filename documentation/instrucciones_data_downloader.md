***

###  `instructions_data_downloader.md`


# 🏛️ Data Core: Ingestion and Market Data Management Pipeline

This module implements an institutional-grade **ETL (Extract, Transform, Load)** architecture designed to build and maintain a local *Data Lake* of financial time series.

Unlike simple download scripts, this core prioritizes **data integrity** (split detection, gap auditing) and **query efficiency** (OLAP via DuckDB).

## 🏗️ System Architecture

The data flow is orchestrated through three sequential stages managed by `master_data_loader.py`:

### 1. Resilient Extraction (`download.py`)
* **Engine:** `yfinance` with custom *wrappers*.
* **Concurrency:** Multi-threaded execution (`ThreadPoolExecutor`) to maximize bandwidth.
* **Incremental Logic:** Automatically detects the last available date on disk and downloads only the necessary *delta*, minimizing network traffic and execution time.
* **Error Handling:** Implements *Exponential Backoff* to retry failed requests without saturating the provider's API.

### 2. Transformation and Audit (`safety.py`)
Acts as a quality *firewall* before data is consumed:
* **Calendar Alignment:** Cross-references each asset against a benchmark index (e.g., `^GSPC`, `^IBEX`) to validate real trading days.
* **Split Detection:** Heuristic algorithm that identifies price drops >30% not explained by the market (benchmark drop >-25%). If detected, it purges the corrupt file to force a clean re-download.
* **Series Repair:** Fills minor gaps via *Forward Fill* and marks these records with `data_quality=0` so models can discriminate them.

### 3. Loading and OLAP Access (`loader.py`)
* **Virtualization:** Does not load all CSV/Parquet files into RAM. Uses **DuckDB** to mount a virtual SQL view over physical files.
* **Zero-Copy:** Transfers query results to **Polars** using Apache Arrow, guaranteeing minimal latency even with millions of rows.

---

## 🔧 Usage Guide: `MarketLoader`

The `MarketLoader` class is the single access interface (Facade) to the Data Lake.

### Initialization

```python
from src.src_DD.loader import MarketLoader

# Inicializa el cargador.
# actualizar_datos=True: Verifica la antigüedad de los datos. Si > 24h, ejecuta el pipeline ETL.
# actualizar_datos=False: Modo "Solo Lectura" (más rápido).
loader = MarketLoader(actualizar_datos=False)