import sys
import os
import polars as pl
import logging
from typing import List, Optional, Union
from datetime import datetime

# --- SETUP DE RUTAS ---
# Calculamos rutas relativas robustas
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/engine/src_features
engine_dir = os.path.dirname(current_dir)                 # src/engine
src_dir = os.path.dirname(engine_dir)                     # src
project_root = os.path.dirname(src_dir)                   # Root del proyecto

# Añadimos al path para importar config
sys.path.append(src_dir)
sys.path.append(engine_dir)

try:
    import pipeline_features
    import config
except ImportError as e:
    print(f"❌ Error crítico de importación en Master: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MASTER_FE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureMaster")

# Definición Robusta de la Ruta de Datos
# Usamos os.path.join desde el root calculado, más seguro que encadenar dirnames
DATA_PATH = os.path.join(project_root, config.PATHS["FEATURES_OUTPUT"])

def _parse_date(date_val: Union[str, datetime]) -> datetime:
    """Helper para asegurar que trabajamos con objetos datetime."""
    if isinstance(date_val, str):
        return datetime.strptime(date_val, "%Y-%m-%d")
    return date_val

def update_features():
    """
    Orquesta la ejecución del Pipeline para regenerar la matriz de features.
    """
    logger.info("🔄 Solicitud de actualización de Features recibida.")
    try:
        # Aquí es donde indirectamente se llama a MarketLoader a través del pipeline
        pipeline_features.run_pipeline()
        logger.info("✅ Actualización completada exitosamente.")
    except Exception as e:
        logger.error(f"❌ Fallo crítico al actualizar features: {e}")
        raise e

def get_feature_matrix(
    tickers: Optional[Union[str, List[str]]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    layer: str = "all" 
) -> pl.DataFrame:
    """
    API PRINCIPAL PARA LOS MODELOS DE ML.
    
    Layers disponibles:
      - 'raw': Indicadores técnicos puros (RSI, MACD...).
      - 'robust': Normalizados temporalmente (Z-Score Robusto).
      - 'neutral': Neutralizados por sector (Cross-Sectional).
      - 'all': Todo lo anterior.
    """
    # Auto-healing: Si no existe el archivo, lo creamos.
    if not os.path.exists(DATA_PATH):
        logger.warning(f"⚠️ No se encontró el archivo en: {DATA_PATH}")
        logger.info("⏳ Ejecutando pipeline por primera vez (Cold Start)...")
        try:
            update_features()
        except Exception:
            logger.error("❌ No se pudo generar la matriz de features.")
            return pl.DataFrame()

    logger.info(f"📂 Cargando matriz de features (Layer: {layer})...")
    
    try:
        # Usamos scan_parquet para carga perezosa (Lazy)
        q = pl.scan_parquet(DATA_PATH)

        # 1. Filtro de Tickers (Push-down optimization)
        if tickers:
            if isinstance(tickers, str):
                tickers = [tickers]
            q = q.filter(pl.col("ticker").is_in(tickers))

        # 2. Filtro de Fechas
        if start_date:
            dt_start = _parse_date(start_date)
            q = q.filter(pl.col("Date") >= pl.lit(dt_start))
        
        if end_date:
            dt_end = _parse_date(end_date)
            q = q.filter(pl.col("Date") <= pl.lit(dt_end))

        # 3. Selección de Columnas (Layering Inteligente)
        if layer != "all":
            # Recolectamos esquema para saber qué columnas existen
            schema = q.collect_schema()
            all_cols = schema.names()
            
            # Columnas base que siempre necesitamos
            meta_cols = {"Date", "ticker", "sector", "country", "data_quality", "Close", "Open", "High", "Low", "Volume"}
            
            # Intersección segura (por si alguna meta no existe)
            selected_cols = list(meta_cols.intersection(set(all_cols)))
            
            if layer == "neutral":
                # Solo las que terminan en _neutral
                selected_cols += [c for c in all_cols if c.endswith("_neutral")]
            elif layer == "robust":
                # Solo las que terminan en _rob
                selected_cols += [c for c in all_cols if c.endswith("_rob")]
            elif layer == "raw":
                # Las que NO son _rob ni _neutral y no son meta
                selected_cols += [c for c in all_cols if c not in meta_cols and not c.endswith("_rob") and not c.endswith("_neutral")]
            
            q = q.select(selected_cols)

        # 4. Materialización Final
        df = q.collect()
        
        if df.is_empty():
            logger.warning("⚠️ La consulta devolvió un DataFrame vacío (revisa filtros de fecha/ticker).")
            
        logger.info(f"✅ Datos recuperados: {df.height} filas, {len(df.columns)} columnas.")
        return df

    except Exception as e:
        logger.error(f"❌ Error leyendo datos: {e}")
        return pl.DataFrame()

if __name__ == "__main__":
    print("\n--- TEST MODO INTERACTIVO MASTER ---")
    # Ejemplo de uso:
    # update_features()
    # df = get_feature_matrix(tickers=["AAPL", "MSFT"], start_date="2023-01-01", layer="neutral")
    # print(df.head())