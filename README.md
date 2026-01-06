# 🏛️ Ben Quant Core

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Architecture](https://img.shields.io/badge/Architecture-Modular_Monolith-blue)

> **Ben Quant Core** es un programa de análisis de activos integral que esta siendo desarrollado de manera activa.

## 📂 Arquitectura del programa:

* **`src/src_DD`**: Bloque de descarga de datos tipo: ETL pipeline utilizando la libreria `yfinance` con backoff exponencial, OLAP storage (DuckDB) y medidas de seguridad.
* **`src/engine/`**:Cálculo de variables, indicadores y modelización.
* **`src/dashboard/`**: Front-end para uso en producción.
* **`src/strategies/`**: Aplicación de los cálculos a métodos de gestión de carteras...
## 🏗️ Arquitectura & ruta de trabajo:

Este proyecto se encuentra en **Fase 1 (Data Core)**. De la siguiente hoja de ruta:

- [x] **Fase 1: Infraestructura de Datos** (Actual)
  - Pipeline ETL tolerante a fallos utilizando `yfinance` con retroceso exponencial (*exponential backoff*).
  - Control de calidad de datos (detección de *splits*, rellenado de huecos...) mediante `safety.py`.
  - Capa de almacenamiento OLAP basada en **DuckDB** y **Parquet**.

- [x] **Fase 2: Ingeniería de Factores** (Próximamente - T1 2026)
  - Motor de indicadores técnicos optimizado con **Polars** y otros.
  - Análisis de microestructura (superficies de volatilidad, *proxies* de liquidez).
  - Otros datos derivados de la modelización.

- [ ] **Fase 3: Modelización de los datos** (T1 2026)
  - Backtester vectorial basado en eventos (*Event-Driven*).
  - Gestión de cartera y métricas de riesgo (Sharpe, Sortino, MaxDD).

- [ ] **Fase 4: Estrategia Alpha** (T2 2026)
  - Integración de los datos recabados en los anteriores bloques para producción.

---

## 🚀 Inicio Rápido (Módulo de Datos)

### Requisitos previos
```bash
pip install -r requirements.txt

