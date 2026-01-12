import sys
import os
import polars as pl
import logging
import time

# --- CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
# Calculamos la raíz del proyecto para importaciones absolutas
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/engine/src_features
engine_dir = os.path.dirname(os.path.dirname(current_dir)) # src/engine
src_dir = os.path.dirname(engine_dir)                     # src
project_root = os.path.dirname(src_dir)                   # Raíz (Benq-Core)

# Añadimos la raíz al path si no está
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # 1. Imports Absolutos (Estándar recomendado)
    # Estos funcionan siempre que project_root esté en el path
    import src.engine.config as config
    import src.engine.src_features.indicators as indicators
    from src.src_DD.loader import MarketLoader
    
except ImportError:
    # 2. Fallback: Configuración legacy de paths
    # Si fallan los absolutos, añadimos rutas específicas e intentamos relativos
    sys.path.append(engine_dir) # Para encontrar 'config'
    sys.path.append(src_dir)    # Para encontrar 'src_DD'
    
    try:
        import config
        import indicators
        from src_DD.loader import MarketLoader
    except ImportError as e:
        print(f"❌ Error CRÍTICO de importación en Pipeline Features: {e}")
        # Debug: Mostrar rutas para diagnosticar
        print(f"Rutas en sys.path: {sys.path}")
        sys.exit(1)


# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PIPELINE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeaturePipeline")

def infer_country(ticker_col: str = "ticker") -> pl.Expr:
    """
    Infiere el código ISO del país basado en el sufijo del Ticker (Yahoo Finance).
    Cubre: Europa, Norteamérica, Asia/Pacífico y Latam.
    """
    t = pl.col(ticker_col)
    
    return (
        # --- EUROPA ---
        pl.when(t.str.ends_with(".DE")).then(pl.lit("DE"))      # Alemania
        .when(t.str.ends_with(".PA")).then(pl.lit("FR"))        # Francia
        .when(t.str.ends_with(".MC")).then(pl.lit("ES"))        # España
        .when(t.str.ends_with(".MI")).then(pl.lit("IT"))        # Italia
        .when(t.str.ends_with(".AS")).then(pl.lit("NL"))        # Holanda (Amsterdam)
        .when(t.str.ends_with(".L")).then(pl.lit("UK"))         # Reino Unido
        .when(t.str.ends_with(".SW")).then(pl.lit("CH"))        # Suiza (Swiss)
        .when(t.str.ends_with(".ST")).then(pl.lit("SE"))        # Suecia (Stockholm)
        .when(t.str.ends_with(".OL")).then(pl.lit("NO"))        # Noruega (Oslo)
        
        # --- ASIA / PACÍFICO ---
        .when(t.str.ends_with(".T")).then(pl.lit("JP"))         # Japón
        .when(t.str.ends_with(".HK")).then(pl.lit("HK"))        # Hong Kong
        .when(t.str.ends_with(".SS").or_(t.str.ends_with(".SZ"))).then(pl.lit("CN")) # China (Shanghai/Shenzhen)
        .when(t.str.ends_with(".KS")).then(pl.lit("KR"))        # Corea (KOSPI)
        .when(t.str.ends_with(".TW")).then(pl.lit("TW"))        # Taiwán
        .when(t.str.ends_with(".AX")).then(pl.lit("AU"))        # Australia
        .when(t.str.ends_with(".NS").or_(t.str.ends_with(".BO"))).then(pl.lit("IN")) # India (NSE/BSE)
        
        # --- AMÉRICA (LATAM + CANADÁ) ---
        .when(t.str.ends_with(".TO").or_(t.str.ends_with(".V"))).then(pl.lit("CA"))  # Canadá (Toronto/Venture)
        .when(t.str.ends_with(".SA")).then(pl.lit("BR"))        # Brasil (Sao Paulo)
        .when(t.str.ends_with(".MX")).then(pl.lit("MX"))        # México
        
        # --- ACTIVOS NO BURSÁTILES ---
        .when(t.str.contains("=")).then(pl.lit("FX/COM"))       # Forex o Commodities
        
        # --- DEFAULT (US) ---
        .otherwise(pl.lit("US"))                                # Sin sufijo suele ser NYSE/NASDAQ
        .alias("country")
    )

def get_robust_zscore(col_name: str, window: int) -> pl.Expr:
    """Z-Score Robusto (Rolling) usando Mediana e IQR."""
    roll_median = pl.col(col_name).rolling_median(window_size=window)
    roll_q75 = pl.col(col_name).rolling_quantile(quantile=0.75, window_size=window, interpolation="linear")
    roll_q25 = pl.col(col_name).rolling_quantile(quantile=0.25, window_size=window, interpolation="linear")
    
    iqr = (roll_q75 - roll_q25)
    
    return (
        (pl.col(col_name) - roll_median) / iqr.replace(0, None)
    ).fill_null(0).alias(f"{col_name}_rob")

def run_pipeline():
    start_total = time.time()
    logger.info("🚀 Iniciando Pipeline de Ingeniería de Features...")

    # 1. CARGA DE DATOS
    logger.info("🔌 Cargando datos desde MarketLoader (DuckDB)...")
    loader = MarketLoader(actualizar_datos=False)
    lf_raw = loader.get_all_data().lazy()

    # 2. PRE-PROCESAMIENTO & CASTING
    logger.info("🌍 Infiriendo metadatos y casteando tipos...")
    lf_base = lf_raw.with_columns([
        infer_country("ticker"),
        pl.col("Date").cast(pl.Datetime),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Float64),
        pl.col("High").cast(pl.Float64),
        pl.col("Low").cast(pl.Float64),
        pl.col("Open").cast(pl.Float64)
    ]).sort(["ticker", "Date"])

    # ---------------------------------------------------------
    # FASE 0: CÁLCULO PREVIO DE RETORNOS (CRÍTICO)
    # ---------------------------------------------------------
    # Necesario para calcular Volatilidad, Skew y Amihud correctamente.
    logger.info("🧮 Fase 0: Pre-calculando Retornos...")
    lf_with_returns = lf_base.with_columns([
        indicators.get_log_returns("Close").over("ticker")
    ]).with_columns([
        # Helper para Amihud
        pl.col("log_returns").abs().alias("abs_log_returns")
    ])

    # ---------------------------------------------------------
    # FASE 1: INDICADORES TÉCNICOS (RAW)
    # ---------------------------------------------------------
    logger.info("🧮 Fase 1: Calculando indicadores técnicos complejos...")
    
    p = config.FEATURES_PARAMS
    
    # 1.1 Indicadores Base (Volatilidad, RSI, MACD...)
    calc_exprs = [
        # --- ESTADÍSTICA DE RETORNOS ---
        indicators.get_rolling_volatility("log_returns", p["VOLATILITY_WINDOW"]),
        indicators.get_rolling_skewness("log_returns", p["SKEW_WINDOW"]),
        indicators.get_volume_return_correlation("log_returns", "Volume", p["CORR_WINDOW"]),
        
        # --- VOLUMEN Y TENDENCIA ---
        indicators.get_volume_std("Volume", p["VOLATILITY_WINDOW"]),
        indicators.get_relative_sma("Close", p["SMA_FAST"]),
        indicators.get_relative_sma("Close", p["SMA_MEDIUM"]),
        indicators.get_relative_sma("Close", p["SMA_SLOW"]),
        indicators.get_efficiency_ratio_ker("Close", p["KER_WINDOW"]),
        *indicators.get_adx("High", "Low", "Close", period=14),
        
        # --- OSCILADORES ---
        indicators.get_rsi("Close", p["RSI_PERIOD"]),
        *indicators.get_macd_expressions("Close", p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"]),

        # --- VOLATILIDAD AVANZADA (Range-Based) ---
        indicators.get_parkinson_volatility("High", "Low", p["PARKINSON_WINDOW"]),
        indicators.get_garman_klass_volatility("High", "Low", "Close", "Open", p["GARMAN_KLASS_WINDOW"]),
        # Yang-Zhang Volatility
        indicators.get_yang_zhang_volatility("Open", "High", "Low", "Close", p["YANG_ZHANG_WINDOW"]),
        
        # --- LIQUIDEZ ---
        indicators.get_amihud_liquidity(
            "abs_log_returns", "Close", "Volume", 
            window=p["AMIHUD_WINDOW"], 
            scaling_factor=p["AMIHUD_SCALING"]
        )
    ]

    lf_indicators_step1 = lf_with_returns.with_columns([
        expr.over("ticker") for expr in calc_exprs
    ])

    # 1.2 Derivados de Volatilidad (Requieren que vol_yz exista)
    # Nombre dinámico de la columna de volatilidad calculada en el paso anterior
    yz_col_name = f"vol_yz_{p['YANG_ZHANG_WINDOW']}d"
    
    lf_indicators = lf_indicators_step1.with_columns([
        *indicators.get_volatility_bounds(
            col_close="Close",
            col_vol_yz=yz_col_name,
            z_score=p["YZ_Z_SCORE"],
            horizon=p["YZ_FORECAST_HORIZON"]
        )
    ])

    # 4. NORMALIZACIÓN TEMPORAL (Robust Scaling por Activo)
    logger.info("📉 Fase 2: Normalización Temporal (Rolling Robust Scaler)...")
    
    schema = lf_indicators.collect_schema()
    
    # Excluimos metadatos y columnas auxiliares de la normalización
    cols_meta = {
        "Date", "ticker", "sector", "country", "data_quality", 
        "Open", "High", "Low", "Close", "Volume", 
        "abs_log_returns" 
    }
    
    # Seleccionamos todo lo demás para normalizar (incluyendo log_returns y los nuevos conos)
    cols_to_normalize = [c for c in schema.names() if c not in cols_meta]
    
    roll_window = config.NORMALIZATION_PARAMS["ROLLING_WINDOW"]
    
    lf_temporal_norm = lf_indicators.with_columns([
        get_robust_zscore(col, roll_window).over("ticker") 
        for col in cols_to_normalize
    ])

    # 5. NEUTRALIZACIÓN SECTORIAL (Cross-Sectional)
    logger.info("⚖️ Fase 3: Neutralización Sectorial (Cross-Sectional)...")
    
    cols_robust = [f"{c}_rob" for c in cols_to_normalize]
    
    neutralization_exprs = []
    for col in cols_robust:
        median_sect = pl.col(col).median()
        q75_sect = pl.col(col).quantile(0.75)
        q25_sect = pl.col(col).quantile(0.25)
        iqr_sect = (q75_sect - q25_sect).fill_null(0) 
        
        expr = ((pl.col(col) - median_sect) / iqr_sect.replace(0, None)).fill_null(0).alias(col.replace("_rob", "_neutral"))
        neutralization_exprs.append(expr)

    lf_final = lf_temporal_norm.with_columns([
        expr.over(["Date", "sector"]) for expr in neutralization_exprs
    ])

    # 6. MATERIALIZACIÓN Y GUARDADO
    logger.info("💾 Materializando y guardando resultados...")
    
    output_path = os.path.join(project_root, config.PATHS["FEATURES_OUTPUT"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        df_result = lf_final.collect()
        
        # Limpieza final: Eliminamos la columna auxiliar de Amihud si existe
        if "abs_log_returns" in df_result.columns:
            df_result = df_result.drop("abs_log_returns")

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