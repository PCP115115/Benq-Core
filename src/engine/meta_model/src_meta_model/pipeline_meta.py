import sys
import os
import polars as pl
import xgboost as xgb
import joblib
import logging
import numpy as np
from datetime import datetime

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.dirname(engine_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import src.engine.config as config
    import src.engine.src_features.indicators as indicators
    from src.engine.meta_model.src_meta_model.download_meta import get_data_meta_model
    
    # --- IMPORTS PARA LA ESTRATEGIA Y PARALELIZACIÓN ---
    import src.strategy.config_strategy as strat_config
    from src.engine.mini_models.src_mini_models.master_mini_models import run_mini_models_pipeline
except ImportError as e:
    print(f"❌ Error de importación en Meta-Model Pipeline: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [META_MODEL] - %(levelname)s - %(message)s')
logger = logging.getLogger("MetaPipeline")

MODEL_DIR = os.path.join(project_root, "src", "data", "models", "meta_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- FUNCIONES DE SOPORTE (TARGET Y ENTRENAMIENTO INDIVIDUAL) ---

def create_dual_target(df: pl.DataFrame, horizon: int, z_score: float, vol_window: int) -> pl.DataFrame:
    """Genera targets basados en Race Logic."""
    # ... (Misma lógica que tenías, resumida para no ocupar espacio innecesario, pero DEBE ESTAR)
    base_vol_col = f"vol_yz_{vol_window}d"
    raw_vol_col = f"{base_vol_col}_RAW"
    target_vol_col = raw_vol_col if raw_vol_col in df.columns else base_vol_col
    
    bounds = indicators.get_volatility_bounds(
        col_close="Close", col_vol_yz=target_vol_col, z_score=z_score, horizon=horizon
    )
    df = df.with_columns(bounds)
    ceil_col = f"fprice_ceil_yz_{horizon}d"
    floor_col = f"fprice_floor_yz_{horizon}d"

    days_to_ceil = []
    days_to_floor = []
    
    for i in range(1, horizon + 1):
        future_high = pl.col("High").shift(-i)
        future_low = pl.col("Low").shift(-i)
        days_to_ceil.append(pl.when(future_high > pl.col(ceil_col)).then(i).otherwise(999))
        days_to_floor.append(pl.when(future_low < pl.col(floor_col)).then(i).otherwise(999))
    
    df = df.with_columns([
        pl.min_horizontal(days_to_ceil).alias("first_ceil_hit"),
        pl.min_horizontal(days_to_floor).alias("first_floor_hit")
    ])

    target_up = ((pl.col("first_ceil_hit") != 999) & (pl.col("first_ceil_hit") < pl.col("first_floor_hit"))).cast(pl.Int8)
    target_down = ((pl.col("first_floor_hit") != 999) & (pl.col("first_floor_hit") < pl.col("first_ceil_hit"))).cast(pl.Int8)
    
    return df.with_columns([target_up.alias("TARGET_UP"), target_down.alias("TARGET_DOWN")]).drop_nulls()

def train_meta_model(ticker: str):
    """Entrena el modelo XGBoost para un solo ticker."""
    logger.info(f"🧠 Iniciando Entrenamiento Meta-Modelo para {ticker}...")
    
    try:
        df_raw = get_data_meta_model(
            ticker=ticker,
            start_date=config.META_MODEL_PARAMS["start_date"],
            end_date=config.META_MODEL_PARAMS["end_date"],
            layer="all",
            feature_list=config.META_MODEL_PARAMS["feature_list"],
            normalization_window=config.META_MODEL_PARAMS["normalization_window"]
        )
    except Exception as e:
        logger.error(f"❌ Fallo al obtener datos: {e}")
        return

    if df_raw.is_empty():
        return

    # Generación del Target
    horizon = config.META_MODEL_PARAMS["FORECAST_HORIZON"]
    z_score = config.FEATURES_PARAMS["YZ_Z_SCORE"]
    vol_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]

    df_dataset = create_dual_target(df_raw, horizon, z_score, vol_window)
    
    exclude_cols = ["Date", "ticker", "sector", "country", "data_quality", "TARGET_UP", "TARGET_DOWN", "first_ceil_hit", "first_floor_hit", f"fprice_ceil_yz_{horizon}d", f"fprice_floor_yz_{horizon}d", "Close", "High", "Low", "Open", "Volume", "log_returns", f"vol_yz_{vol_window}d_RAW"]
    feature_cols = [c for c in df_dataset.columns if c not in exclude_cols]

    train_params = config.META_MODEL_PARAMS["META_MODEL_TRAIN_PARAMS"]
    split_ratio = train_params["TRAIN_TEST_SPLIT_RATIO"]
    X = df_dataset.select(feature_cols).to_pandas()
    y_up = df_dataset.select("TARGET_UP").to_pandas().values.ravel()
    y_down = df_dataset.select("TARGET_DOWN").to_pandas().values.ravel()

    cutoff_idx = int(len(X) * split_ratio)
    train_end_idx = cutoff_idx - (horizon if train_params["PURGE_OVERLAP"] else 0)

    if train_params["TEST_MODE"]:
        X_train, X_test = X.iloc[:train_end_idx], X.iloc[cutoff_idx:]
        y_train_up, y_test_up = y_up[:train_end_idx], y_up[cutoff_idx:]
        y_train_down, y_test_down = y_down[:train_end_idx], y_down[cutoff_idx:]
    else:
        X_train, X_test = X, None
        y_train_up, y_test_up = y_up, None
        y_train_down, y_test_down = y_down, None

    targets_config = [("UP", y_train_up, y_test_up), ("DOWN", y_train_down, y_test_down)]
    base_xgb_params = config.META_MODEL_PARAMS["XGB_PARAMS"].copy()

    for direction, y_train, y_test in targets_config:
        n_pos = np.sum(y_train)
        if n_pos > 0:
            base_xgb_params["scale_pos_weight"] = (len(y_train) - n_pos) / n_pos
        
        model = xgb.XGBClassifier(**base_xgb_params)
        eval_set = [(X_train, y_train)]
        if train_params["TEST_MODE"]: eval_set.append((X_test, y_test))

        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        path = os.path.join(MODEL_DIR, f"xgboost_meta_{direction.lower()}.json")
        model.save_model(path)
        logger.info(f"   💾 Modelo guardado: {path}")

# --- LA NUEVA FUNCIÓN OPTIMIZADA ---

def run_batch_training():
    """
    Orquesta el entrenamiento masivo para aprovechar todos los núcleos del PC.
    """
    start_global = datetime.now()
    tickers = strat_config.TICKERS_ESTRATEGIA
    
    logger.info("="*60)
    logger.info(f"🚀 INICIANDO ENTRENAMIENTO BATCH PARA {len(tickers)} TICKERS")
    logger.info("="*60)

    # PASO 1: GENERACIÓN MASIVA DE DATOS (Vectorizado - 15 Núcleos)
    logger.info(f"⚡ FASE 1: Ejecutando Mini-Modelos en Paralelo (n_jobs=-1)...")
    run_mini_models_pipeline(tickers, n_jobs=-1)
    
    # PASO 2: ENTRENAMIENTO DE META-MODELOS (Secuencial - XGBoost rápido)
    logger.info(f"🧠 FASE 2: Entrenando Meta-Modelos (XGBoost)...")
    for ticker in tickers:
        try:
            train_meta_model(ticker)
        except Exception as e:
            logger.error(f"❌ Error entrenando {ticker}: {e}")

    elapsed = datetime.now() - start_global
    logger.info("="*60)
    logger.info(f"🏁 ENTRENAMIENTO BATCH COMPLETADO EN {elapsed.total_seconds():.2f}s")
    logger.info("="*60)

if __name__ == "__main__":
    run_batch_training()