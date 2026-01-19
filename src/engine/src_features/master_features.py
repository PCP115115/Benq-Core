import sys
import os
import polars as pl
import logging
from typing import List, Optional, Union
from datetime import datetime

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/engine/src_features
engine_dir = os.path.dirname(current_dir)                 # src/engine
src_dir = os.path.dirname(engine_dir)                     # src
project_root = os.path.dirname(src_dir)                   # Root del proyecto

# --- CORRECCIÓN DE IMPORTS (LO ÚNICO QUE CAMBIA) ---
# Añadimos la raíz al path si no está, para que Python encuentre 'src'
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # 1. Intentamos Import Absoluto (Correcto para tests y ejecución desde raíz)
    import src.engine.config as config
    import src.engine.src_features.pipeline_features as pipeline_features
except ImportError:
    # 2. Fallback: Import Local (Correcto para ejecución directa del script)
    sys.path.append(engine_dir)
    sys.path.append(current_dir)
    try:
        import config
        import pipeline_features
    except ImportError as e:
        print(f"❌ Error crítico de importación en Master: {e}")
        sys.exit(1)
# ---------------------------------------------------
# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MASTER_FE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureMaster")

# Definición Robusta de la Ruta de Datos
DATA_PATH = os.path.join(project_root, config.PATHS["FEATURES_OUTPUT"])

def _parse_date(date_val: Union[str, datetime]) -> datetime:
    """Helper para asegurar que trabajamos con objetos datetime standard."""
    if isinstance(date_val, str):
        return datetime.strptime(date_val, "%Y-%m-%d")
    return date_val

def update_features():
    """
    Orquesta la ejecución del Pipeline para regenerar la matriz de features.
    Garantiza la consistencia de los datos en disco (Cold Start / Refresh).
    """
    logger.info("🔄 Solicitud de actualización de Features recibida.")
    try:
        pipeline_features.run_pipeline()
        logger.info("✅ Actualización completada exitosamente.")
    except Exception as e:
        logger.error(f"❌ Fallo crítico al actualizar features: {e}")
        raise e

def get_feature_matrix(
    tickers: Optional[Union[str, List[str]]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    layer: str = "all",
    features: Optional[Union[str, List[str]]] = None,
    normalization_window: Optional[int] = None  # <--- NUEVO PARÁMETRO
) -> pl.DataFrame:
    """
    API PRINCIPAL PARA CONSUMO DE DATOS EN MODELOS QUANT.
    
    Permite filtrar por dimensiones de tiempo, activo (tickers), capa de procesamiento (layer)
    y características específicas (features).

    Args:
        tickers: Ticker individual o lista de tickers.
        start_date: Fecha de inicio (inclusiva).
        end_date: Fecha de fin (inclusiva).
        layer: Nivel de procesamiento ('raw', 'robust', 'neutral', 'all').
        features: Filtro de columnas por palabra clave.
        normalization_window: Si se define (int), aplica normalización robusta al vuelo 
                              con esa ventana, ignorando 'layer'.

    Returns:
        pl.DataFrame: Matriz de features filtrada.
    """
    
    # 1. Auto-healing: Cold Start Check
    if not os.path.exists(DATA_PATH):
        logger.warning(f"⚠️ No se encontró el archivo en: {DATA_PATH}")
        logger.info("⏳ Ejecutando pipeline por primera vez (Cold Start)...")
        try:
            update_features()
        except Exception:
            logger.error("❌ No se pudo generar la matriz de features.")
            return pl.DataFrame()

    logger.info(f"📂 Cargando matriz (Layer: {layer} | Features: {features} | NormWindow: {normalization_window})...")
    
    try:
        # Iniciamos LazyFrame para Push-down optimization
        q = pl.scan_parquet(DATA_PATH)

        # -----------------------------------------------------------
        # 2. OPTIMIZACIÓN DE FILTROS (ROWS)
        # -----------------------------------------------------------
        if tickers:
            if isinstance(tickers, str):
                tickers = [tickers]
            q = q.filter(pl.col("ticker").is_in(tickers))

        if start_date:
            dt_start = _parse_date(start_date)
            q = q.filter(pl.col("Date") >= pl.lit(dt_start))
        
        if end_date:
            dt_end = _parse_date(end_date)
            q = q.filter(pl.col("Date") <= pl.lit(dt_end))

        # -----------------------------------------------------------
        # 3. LÓGICA DE SELECCIÓN DE COLUMNAS
        # -----------------------------------------------------------
        
        # === MODO A: NORMALIZACIÓN DINÁMICA (NUEVO) ===
        if normalization_window is not None:
            # 1. Identificar columnas RAW objetivo
            target_keywords = []
            if features and features != "all":
                if isinstance(features, str): target_keywords = [features]
                else: target_keywords = features
            
            schema = q.collect_schema()
            all_cols = schema.names()
            
            # Definir columnas meta y excluidas
            excluded_suffixes = ("_rob", "_neutral")
            meta_cols = {"Date", "ticker", "sector", "country", "data_quality", "Close", "Open", "High", "Low", "Volume"}
            
            # Seleccionar columnas que sean RAW y coincidan con keywords (si las hay)
            target_cols = []
            for col in all_cols:
                if col in meta_cols: continue
                if col.endswith(excluded_suffixes): continue
                
                # Si hay filtro de features, debe coincidir
                if target_keywords:
                    if not any(kw in col for kw in target_keywords):
                        continue
                
                target_cols.append(col)

            if not target_cols:
                logger.warning("⚠️ No se encontraron columnas RAW para normalizar con los filtros dados.")
                return pl.DataFrame()

            # 2. Seleccionar (Date/Ticker + Targets)
            q = q.select(["Date", "ticker"] + target_cols)

            # 3. Aplicar Robust Scaler Rolling por Ticker
            # Z = (X - Median) / IQR
            exprs_norm = []
            for col in target_cols:
                roll_med = pl.col(col).rolling_median(normalization_window)
                
                # --- CAMBIO AQUÍ: Añadimos interpolation='linear' ---
                roll_q75 = pl.col(col).rolling_quantile(
                    0.75, 
                    window_size=normalization_window, 
                    interpolation='linear'  # <--- CRÍTICO PARA COINCIDIR CON NUMPY/PANDAS
                )
                roll_q25 = pl.col(col).rolling_quantile(
                    0.25, 
                    window_size=normalization_window, 
                    interpolation='linear'  # <--- CRÍTICO
                )
                # ----------------------------------------------------

                iqr = (roll_q75 - roll_q25).replace(0, None) # Evitar div/0 (NULL si IQR es 0)
                
                # Sobrescribimos la columna con su versión normalizada y rellenamos nulos con 0
                expr = ((pl.col(col) - roll_med) / iqr).fill_null(0).alias(col)
                exprs_norm.append(expr)
                
            q = q.with_columns([e.over("ticker") for e in exprs_norm])
            
            # 4. Limpieza de NaNs iniciales (periodo de calentamiento)
            q = q.drop_nulls()

        # === MODO B: LÓGICA ESTÁNDAR (ORIGINAL) ===
        else:
            # Obtenemos esquema para determinar disponibilidad
            schema = q.collect_schema()
            all_cols = schema.names()
            
            # Definición de conjuntos de metadatos
            MANDATORY_COLS = {"Date", "ticker"} 
            META_COLS = {"sector", "country", "data_quality", "Close", "Open", "High", "Low", "Volume"}
            ALL_META = MANDATORY_COLS.union(META_COLS)

            # A. Determinar el Universo disponible según LAYER
            candidates = []
            if layer == "all":
                candidates = all_cols
            else:
                neutral_cols = [c for c in all_cols if c.endswith("_neutral")]
                robust_cols = [c for c in all_cols if c.endswith("_rob")]
                raw_feature_cols = [
                    c for c in all_cols 
                    if c not in ALL_META 
                    and not c.endswith("_neutral") 
                    and not c.endswith("_rob")
                ]
                
                base_meta = [c for c in all_cols if c in ALL_META]
                
                if layer == "neutral":
                    candidates = base_meta + neutral_cols
                elif layer == "robust":
                    candidates = base_meta + robust_cols
                elif layer == "raw":
                    candidates = base_meta + raw_feature_cols

            # B. Aplicar filtro de FEATURES (Intersección)
            final_columns = []
            feature_filter_active = False
            target_keywords = []

            if features is not None and features != "all":
                feature_filter_active = True
                if isinstance(features, str):
                    target_keywords = [features]
                else:
                    target_keywords = features

            if not feature_filter_active:
                final_columns = candidates
            else:
                matched_count = 0
                for col in candidates:
                    if col in MANDATORY_COLS:
                        final_columns.append(col)
                        continue
                    
                    is_match = any(kw in col for kw in target_keywords)
                    if is_match:
                        final_columns.append(col)
                        matched_count += 1
                
                if matched_count == 0:
                    error_msg = (
                        f"❌ Filtro inválido: No se encontraron columnas en la layer '{layer}' "
                        f"que coincidan con los criterios: {target_keywords}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            # Aplicar proyección
            q = q.select(final_columns)

        # -----------------------------------------------------------
        # 4. MATERIALIZACIÓN Y RETORNO
        # -----------------------------------------------------------
        df = q.collect()
        
        if df.is_empty():
            logger.warning("⚠️ La consulta devolvió un DataFrame vacío (revisa filtros de fecha/ticker).")
        else:
            logger.info(f"✅ Datos recuperados: {df.height} filas, {len(df.columns)} columnas.")

        return df

    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.error(f"❌ Error leyendo datos: {e}")
        return pl.DataFrame()

if __name__ == "__main__":
    try:
        from src.strategy.config_strategy import TICKERS_ESTRATEGIA
        df = get_feature_matrix(tickers=TICKERS_ESTRATEGIA, layer="robust", features="all")
        print(df.head())
    except Exception as e:
        logger.error(f"❌ Error en ejecución directa: {e}")

