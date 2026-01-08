import sys
import os
import polars as pl
import logging
from typing import List, Optional, Union
from datetime import datetime

# --- SETUP DE RUTAS ---
# Calculamos rutas relativas robustas para mantener la modularidad del sistema
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/engine/src_features
engine_dir = os.path.dirname(current_dir)                 # src/engine
src_dir = os.path.dirname(engine_dir)                     # src
project_root = os.path.dirname(src_dir)                   # Root del proyecto

# Añadimos al path para importar config y módulos hermanos
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
    features: Optional[Union[str, List[str]]] = None
) -> pl.DataFrame:
    """
    API PRINCIPAL PARA CONSUMO DE DATOS EN MODELOS QUANT.
    
    Permite filtrar por dimensiones de tiempo, activo (tickers), capa de procesamiento (layer)
    y características específicas (features).

    Args:
        tickers: Ticker individual o lista de tickers.
        start_date: Fecha de inicio (inclusiva).
        end_date: Fecha de fin (inclusiva).
        layer: Nivel de procesamiento:
               - 'raw': Indicadores técnicos puros.
               - 'robust': Normalizados temporalmente (Rolling Z-Score).
               - 'neutral': Neutralizados por sector.
               - 'all': Universo completo.
        features: Filtro de columnas por palabra clave (substring). 
                  Ej: ["rsi", "volatility"]. Si es None o "all", devuelve todo lo de la layer.
                  NOTA: 'Date' y 'ticker' siempre se devuelven.

    Returns:
        pl.DataFrame: Matriz de features filtrada.
    
    Raises:
        ValueError: Si el filtro de 'features' no produce ninguna columna válida en la 'layer' seleccionada.
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

    logger.info(f"📂 Cargando matriz de features (Layer: {layer} | Features Filter: {features})...")
    
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
        # 3. LÓGICA DE SELECCIÓN DE COLUMNAS (INTERSECCIÓN LAYER + FEATURES)
        # -----------------------------------------------------------
        
        # Obtenemos esquema para determinar disponibilidad
        schema = q.collect_schema()
        all_cols = schema.names()
        
        # Definición de conjuntos de metadatos
        # Date y ticker son "Intocables" (Primary Keys)
        MANDATORY_COLS = {"Date", "ticker"} 
        # Otros metadatos (pueden ser filtrados si el usuario pide features específicos)
        META_COLS = {"sector", "country", "data_quality", "Close", "Open", "High", "Low", "Volume"}
        ALL_META = MANDATORY_COLS.union(META_COLS)

        # A. Determinar el Universo disponible según LAYER
        candidates = []
        if layer == "all":
            candidates = all_cols
        else:
            # Clasificación de columnas por sufijo
            neutral_cols = [c for c in all_cols if c.endswith("_neutral")]
            robust_cols = [c for c in all_cols if c.endswith("_rob")]
            # Raw son las que no tienen sufijos de procesamiento y no son meta (o son meta base)
            raw_feature_cols = [
                c for c in all_cols 
                if c not in ALL_META 
                and not c.endswith("_neutral") 
                and not c.endswith("_rob")
            ]
            
            # Construimos el set base de la layer (incluyendo meta por defecto)
            # Nota: La meta se refina después si hay filtro de features
            base_meta = [c for c in all_cols if c in ALL_META]
            
            if layer == "neutral":
                candidates = base_meta + neutral_cols
            elif layer == "robust":
                candidates = base_meta + robust_cols
            elif layer == "raw":
                candidates = base_meta + raw_feature_cols

        # B. Aplicar filtro de FEATURES (Intersección)
        final_columns = []
        
        # Normalizamos input de features
        feature_filter_active = False
        target_keywords = []

        if features is not None and features != "all":
            feature_filter_active = True
            if isinstance(features, str):
                target_keywords = [features]
            else:
                target_keywords = features

        if not feature_filter_active:
            # Si no hay filtro específico, devolvemos todo lo que permite la layer
            final_columns = candidates
        else:
            # LÓGICA STRICT DE FILTRADO
            # 1. Siempre incluir MANDATORY (Date, ticker)
            # 2. Incluir columna SI Y SOLO SI contiene alguna keyword solicitada
            
            matched_count = 0
            
            for col in candidates:
                # Condición 1: Es obligatoria
                if col in MANDATORY_COLS:
                    final_columns.append(col)
                    continue
                
                # Condición 2: Match por substring
                # "rsi" debe estar en "rsi_14_neutral" -> True
                # "rsi" en "Close" -> False (Close se elimina si no se pidió explícitamente)
                is_match = any(kw in col for kw in target_keywords)
                
                if is_match:
                    final_columns.append(col)
                    matched_count += 1
            
            # CHECK DE SEGURIDAD (Safety)
            if matched_count == 0:
                error_msg = (
                    f"❌ Filtro inválido: No se encontraron columnas en la layer '{layer}' "
                    f"que coincidan con los criterios: {target_keywords}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Aplicar proyección (Push-down projection)
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
        # Re-lanzamos ValueErrors (validación de lógica de negocio)
        raise ve
    except Exception as e:
        logger.error(f"❌ Error leyendo datos: {e}")
        # En caso de error de I/O u otro imprevisto, devolvemos vacío para no romper procesos batch masivos
        # salvo que sea un error de lógica de filtrado (ValueError arriba)
        return pl.DataFrame()

if __name__ == "__main__":
    print("\n--- TEST MODO INTERACTIVO MASTER ---")
    # Bloque de prueba manual
    # try:
    #     df = get_feature_matrix(
    #         tickers=["AAPL", "MSFT"], 
    #         layer="neutral", 
    #         features=["rsi", "volatility"]
    #     )
    #     print(df.head())
    #     print("Columnas:", df.columns)
    # except Exception as e:
    #     print(e)