import sys
import os
import polars as pl
import logging
from typing import List, Optional, Union
from datetime import datetime

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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

# Ruta absoluta al archivo parquet de salida
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), config.PATHS["FEATURES_OUTPUT"])

def _parse_date(date_val: Union[str, datetime]) -> datetime:
    """Helper para asegurar que trabajamos con objetos datetime, no strings."""
    if isinstance(date_val, str):
        # Asumimos formato YYYY-MM-DD
        return datetime.strptime(date_val, "%Y-%m-%d")
    return date_val

def update_features():
    """
    Fuerza la actualización completa de la matriz de features.
    """
    logger.info("🔄 Solicitud de actualización de Features recibida.")
    try:
        pipeline_features.run_pipeline()
        logger.info("✅ Actualización completada.")
    except Exception as e:
        logger.error(f"❌ Fallo al actualizar features: {e}")
        raise e

def get_feature_matrix(
    tickers: Optional[Union[str, List[str]]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    layer: str = "all" 
) -> pl.DataFrame:
    """
    API PRINCIPAL PARA MODELOS.
    """
    if not os.path.exists(DATA_PATH):
        logger.warning(f"⚠️ No se encontró el archivo de features en: {DATA_PATH}")
        logger.info("⏳ Ejecutando pipeline por primera vez...")
        try:
            update_features()
        except Exception:
            # Si falla la actualización (ej. en test simulado), retornamos vacío para evitar crash
            return pl.DataFrame()

    logger.info(f"📂 Cargando matriz de features (Layer: {layer})...")
    
    try:
        q = pl.scan_parquet(DATA_PATH)

        # 1. Filtro de Tickers
        if tickers:
            if isinstance(tickers, str):
                tickers = [tickers]
            q = q.filter(pl.col("ticker").is_in(tickers))

        # 2. Filtro de Fechas (CORREGIDO: Casting explícito a datetime)
        if start_date:
            dt_start = _parse_date(start_date)
            q = q.filter(pl.col("Date") >= pl.lit(dt_start))
        
        if end_date:
            dt_end = _parse_date(end_date)
            q = q.filter(pl.col("Date") <= pl.lit(dt_end))

        # 3. Selección de Columnas (Layering)
        if layer != "all":
            schema = q.collect_schema()
            all_cols = schema.names()
            
            meta_cols = {"Date", "ticker", "sector", "country", "data_quality", "Close", "Open", "High", "Low", "Volume"}
            
            selected_cols = list(meta_cols.intersection(set(all_cols)))
            
            if layer == "neutral":
                selected_cols += [c for c in all_cols if c.endswith("_neutral")]
            elif layer == "robust":
                selected_cols += [c for c in all_cols if c.endswith("_rob")]
            elif layer == "raw":
                # Raw = No termina en _rob ni _neutral y no es meta
                selected_cols += [c for c in all_cols if c not in meta_cols and not c.endswith("_rob") and not c.endswith("_neutral")]
            
            q = q.select(selected_cols)

        # 4. Materialización
        df = q.collect()
        logger.info(f"✅ Datos recuperados: {df.height} filas, {len(df.columns)} columnas.")
        return df

    except Exception as e:
        logger.error(f"❌ Error leyendo datos: {e}")
        # Devolvemos DataFrame vacío en caso de error para no romper el flujo del programa
        return pl.DataFrame()

if __name__ == "__main__":
    print("\n--- TEST MODO INTERACTIVO MASTER ---")
    # update_features()
    # df = get_feature_matrix(tickers=None, start_date="2023-01-01", layer="neutral")