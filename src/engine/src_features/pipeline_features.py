import sys
import os
import polars as pl
import logging
import time

# --- CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(current_dir)) # src/engine
src_dir = os.path.dirname(engine_dir)                    # src
project_root = os.path.dirname(src_dir)                  # Root

# Añadimos rutas al path para importar módulos hermanos
sys.path.append(src_dir) 
sys.path.append(engine_dir)
# Para importar indicators que está en la misma carpeta
sys.path.append(current_dir)

try:
    from src_DD.loader import MarketLoader
    import config
    import indicators
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PIPELINE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeaturePipeline")

def infer_country(ticker_col: str = "ticker") -> pl.Expr:
    """
    Infiere el país basado en el sufijo de Yahoo Finance.
    Esto permite neutralizar por geografía sin tener el dato explícito.
    """
    return (
        pl.when(pl.col(ticker_col).str.ends_with(".MC")).then(pl.lit("ES"))
        .when(pl.col(ticker_col).str.ends_with(".DE")).then(pl.lit("DE"))
        .when(pl.col(ticker_col).str.ends_with(".L")).then(pl.lit("UK"))
        .when(pl.col(ticker_col).str.ends_with(".PA")).then(pl.lit("FR"))
        .when(pl.col(ticker_col).str.ends_with(".T")).then(pl.lit("JP"))
        .when(pl.col(ticker_col).str.ends_with(".HK")).then(pl.lit("HK"))
        .when(pl.col(ticker_col).str.contains("=")).then(pl.lit("FX/COM")) # Futuros/Forex
        .otherwise(pl.lit("US")) # Por defecto, sin sufijo es USA
        .alias("country")
    )

def get_robust_zscore(col_name: str, window: int) -> pl.Expr:
    """
    Calcula el Z-Score Robusto (basado en Mediana e IQR) de forma RODANTE.
    Z_Robust = (X - Median) / (Q75 - Q25)
    
    Ventaja: Los outliers extremos no 'aplastan' la distribución.
    """
    roll_median = pl.col(col_name).rolling_median(window_size=window)
    roll_q75 = pl.col(col_name).rolling_quantile(quantile=0.75, window_size=window, interpolation="linear")
    roll_q25 = pl.col(col_name).rolling_quantile(quantile=0.25, window_size=window, interpolation="linear")
    
    iqr = (roll_q75 - roll_q25)
    
    # Evitamos división por cero reemplazando 0 por null o un epsilon muy pequeño
    # Si IQR es 0 (precio plano), el z-score debería ser 0.
    return (
        (pl.col(col_name) - roll_median) / iqr.replace(0, None)
    ).fill_null(0).alias(f"{col_name}_rob")

def run_pipeline():
    start_total = time.time()
    logger.info("🚀 Iniciando Pipeline de Ingeniería de Features...")

    # 1. CARGA DE DATOS (MarketLoader)
    logger.info("🔌 Cargando datos desde MarketLoader (DuckDB)...")
    loader = MarketLoader(actualizar_datos=False)
    
    # Usamos LazyFrame para máxima eficiencia de memoria
    lf_raw = loader.get_all_data().lazy()

    # 2. PRE-PROCESAMIENTO & METADATA
    logger.info("🌍 Infiriendo metadatos (País)...")
    lf_base = lf_raw.with_columns([
        infer_country("ticker"),
        pl.col("Date").cast(pl.Datetime),
        # Aseguramos tipos numéricos para cálculos
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Float64)
    ]).sort(["ticker", "Date"])

    # 3. CÁLCULO DE INDICADORES (RAW)
    # Aquí llamamos a indicators.py. Usamos over("ticker") para calcular por activo.
    logger.info("🧮 Calculando indicadores técnicos (Fase 1: Raw)...")
    
    p = config.FEATURES_PARAMS
    
    # Definimos la lista de expresiones a calcular
    # MACD devuelve una lista, así que usamos * para desempaquetarla
    calc_exprs = [
        indicators.get_log_returns("Close"),
        indicators.get_rolling_volatility("Close", p["VOLATILITY_WINDOW"]),
        indicators.get_volume_std("Volume", p["VOLATILITY_WINDOW"]),
        indicators.get_rolling_skewness("Close", p["SKEW_WINDOW"]),
        indicators.get_volume_return_correlation("Close", "Volume", p["CORR_WINDOW"]),
        indicators.get_relative_sma("Close", p["SMA_FAST"]),
        indicators.get_relative_sma("Close", p["SMA_MEDIUM"]),
        indicators.get_relative_sma("Close", p["SMA_SLOW"]),
        indicators.get_efficiency_ratio_ker("Close", p["KER_WINDOW"]),
        indicators.get_parkinson_volatility("High", "Low", p["PARKINSON_WINDOW"]),
        indicators.get_rsi("Close", p["RSI_PERIOD"]),
        indicators.get_amihud_liquidity("Close", "Close", "Volume", p["AMIHUD_WINDOW"]),
        *indicators.get_macd_expressions("Close", p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    ]

    lf_indicators = lf_base.with_columns([
        expr.over("ticker") for expr in calc_exprs
    ])

    # 4. NORMALIZACIÓN TEMPORAL (Robust Scaling por Activo)
    # Comparamos cada activo consigo mismo en el pasado
    logger.info("📉 Fase 2: Normalización Temporal (Rolling Robust Scaler)...")
    
    # Identificamos las columnas numéricas que acabamos de crear (excluyendo OHLCV y meta)
    # Nota: Como es Lazy, necesitamos una lista de nombres. 
    # Por seguridad, definimos explícitamente qué normalizar o lo inferimos después.
    # Estrategia: Normalizamos TODO lo que no sea meta.
    
    # Para poder listar columnas, hacemos un fetch de esquema (muy rápido)
    schema = lf_indicators.collect_schema()
    cols_meta = {"Date", "ticker", "sector", "country", "data_quality", "Open", "High", "Low", "Close", "Volume"}
    cols_to_normalize = [c for c in schema.names() if c not in cols_meta]
    
    roll_window = config.NORMALIZATION_PARAMS["ROLLING_WINDOW"]
    
    # Aplicamos Robust Z-Score sobre cada ticker
    lf_temporal_norm = lf_indicators.with_columns([
        get_robust_zscore(col, roll_window).over("ticker") 
        for col in cols_to_normalize
    ])

    # 5. NEUTRALIZACIÓN SECTORIAL (Cross-Sectional)
    # Agrupamos por Fecha y Sector para eliminar el riesgo de mercado/sector
    logger.info("⚖️ Fase 3: Neutralización Sectorial (Cross-Sectional)...")
    
    # Seleccionamos solo las columnas ya normalizadas temporalmente (las que terminan en _rob)
    cols_robust = [f"{c}_rob" for c in cols_to_normalize]
    
    # Fórmula: (Valor - Mediana_Sector) / IQR_Sector
    # Esto se hace por FECHA y SECTOR.
    
    neutralization_exprs = []
    for col in cols_robust:
        median_sect = pl.col(col).median()
        q75_sect = pl.col(col).quantile(0.75)
        q25_sect = pl.col(col).quantile(0.25)
        iqr_sect = (q75_sect - q25_sect).fill_null(0) # Evitar nulos
        
        # Nombre final: feature_neutral
        expr = ((pl.col(col) - median_sect) / iqr_sect.replace(0, None)).fill_null(0).alias(col.replace("_rob", "_neutral"))
        neutralization_exprs.append(expr)

    # Aplicamos la neutralización agrupando por Fecha y Sector
    lf_final = lf_temporal_norm.with_columns([
        expr.over(["Date", "sector"]) for expr in neutralization_exprs
    ])

    # 6. MATERIALIZACIÓN Y GUARDADO
    logger.info("💾 Materializando y guardando resultados...")
    
    # Filtramos los primeros N días donde los rolling windows son nulos
    min_periods = config.NORMALIZATION_PARAMS["ROLLING_WINDOW"]
    
    # Ruta de salida relativa al project root
    output_path = os.path.join(project_root, config.PATHS["FEATURES_OUTPUT"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Collect final: Aquí es donde Polars ejecuta todo el grafo
        df_result = lf_final.collect()
        
        # Opcional: Eliminar filas iniciales con muchos nulos por los windows
        # df_result = df_result.filter(pl.col("Date") > pl.col("Date").min() + pl.duration(days=min_periods))
        
        df_result.write_parquet(output_path)
        
        rows = df_result.height
        cols = len(df_result.columns)
        size_mb = df_result.estimated_size("mb")
        
        logger.info(f"✅ PIPELINE FINALIZADO EXITOSAMENTE")
        logger.info(f"   - Dimensiones: {rows:,} filas x {cols} columnas")
        logger.info(f"   - Tamaño: {size_mb:.2f} MB")
        logger.info(f"   - Archivo: {output_path}")
        logger.info(f"   - Tiempo Total: {time.time() - start_total:.2f}s")
        
    except Exception as e:
        logger.critical(f"❌ Error durante la ejecución del pipeline: {e}")
        raise e

if __name__ == "__main__":
    run_pipeline()
