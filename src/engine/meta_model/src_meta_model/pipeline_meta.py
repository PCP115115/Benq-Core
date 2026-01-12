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
except ImportError as e:
    print(f"❌ Error de importación en Meta-Model Pipeline: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [META_MODEL] - %(levelname)s - %(message)s')
logger = logging.getLogger("MetaPipeline")

MODEL_DIR = os.path.join(project_root, "src", "data", "models", "meta_model")
os.makedirs(MODEL_DIR, exist_ok=True)

def create_dual_target(df: pl.DataFrame, horizon: int, z_score: float, vol_window: int) -> pl.DataFrame:
    """
    Genera DOS targets basados en 'Race Logic'.
    Busca automáticamente la volatilidad RAW para el cálculo correcto de barreras.
    """
    logger.info("🎯 Calculando Lógica de Carrera (Dual Breakout) para el Target...")
    
    base_vol_col = f"vol_yz_{vol_window}d"
    raw_vol_col = f"{base_vol_col}_RAW"
    target_vol_col = raw_vol_col if raw_vol_col in df.columns else base_vol_col
    
    required_cols = ["Close", "High", "Low", target_vol_col]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        raise ValueError(f"Faltan columnas para calcular el Target: {missing}. Revisa download_meta.py")

    bounds = indicators.get_volatility_bounds(
        col_close="Close",
        col_vol_yz=target_vol_col,
        z_score=z_score,
        horizon=horizon
    )
    df = df.with_columns(bounds)
    
    ceil_col = f"fprice_ceil_yz_{horizon}d"
    floor_col = f"fprice_floor_yz_{horizon}d"

    days_to_ceil = []
    days_to_floor = []
    
    for i in range(1, horizon + 1):
        future_high = pl.col("High").shift(-i)
        future_low = pl.col("Low").shift(-i)
        
        hit_c = pl.when(future_high > pl.col(ceil_col)).then(i).otherwise(999)
        hit_f = pl.when(future_low < pl.col(floor_col)).then(i).otherwise(999)
        
        days_to_ceil.append(hit_c)
        days_to_floor.append(hit_f)
    
    df = df.with_columns([
        pl.min_horizontal(days_to_ceil).alias("first_ceil_hit"),
        pl.min_horizontal(days_to_floor).alias("first_floor_hit")
    ])

    target_up = ((pl.col("first_ceil_hit") != 999) & (pl.col("first_ceil_hit") < pl.col("first_floor_hit"))).cast(pl.Int8)
    target_down = ((pl.col("first_floor_hit") != 999) & (pl.col("first_floor_hit") < pl.col("first_ceil_hit"))).cast(pl.Int8)
    
    return df.with_columns([
        target_up.alias("TARGET_UP"),
        target_down.alias("TARGET_DOWN")
    ]).drop_nulls()

def train_meta_model(ticker: str = config.ticker_analizado):
    start_time = datetime.now()
    logger.info(f"🧠 Iniciando Pipeline Meta-Modelo Dual para {ticker}...")
    
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
        logger.error("❌ El DataFrame está vacío.")
        return

    # --- GENERACIÓN DEL TARGET ---
    horizon = config.META_MODEL_PARAMS["FORECAST_HORIZON"]
    z_score = config.FEATURES_PARAMS["YZ_Z_SCORE"]
    vol_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]

    df_dataset = create_dual_target(df_raw, horizon, z_score, vol_window)
    
    exclude_cols = [
        "Date", "ticker", "sector", "country", "data_quality",
        "TARGET_UP", "TARGET_DOWN", 
        "first_ceil_hit", "first_floor_hit", 
        f"fprice_ceil_yz_{horizon}d", f"fprice_floor_yz_{horizon}d",
        "Close", "High", "Low", "Open", "Volume", "log_returns",
        f"vol_yz_{vol_window}d_RAW"
    ]
    
    feature_cols = [c for c in df_dataset.columns if c not in exclude_cols]
    logger.info(f"📊 Features de Entrenamiento ({len(feature_cols)}): {feature_cols}")

    train_params = config.META_MODEL_PARAMS["META_MODEL_TRAIN_PARAMS"]
    split_ratio = train_params["TRAIN_TEST_SPLIT_RATIO"]
    test_mode = train_params["TEST_MODE"]
    purge_overlap = train_params["PURGE_OVERLAP"]

    X = df_dataset.select(feature_cols).to_pandas()
    y_up = df_dataset.select("TARGET_UP").to_pandas().values.ravel()
    y_down = df_dataset.select("TARGET_DOWN").to_pandas().values.ravel()

    cutoff_idx = int(len(X) * split_ratio)
    purge_size = horizon if purge_overlap else 0
    train_end_idx = cutoff_idx - purge_size

    if test_mode:
        logger.info(f"🧪 MODO TEST: Split {split_ratio:.0%} | Purge Gap: {purge_size} días")
        X_train, X_test = X.iloc[:train_end_idx], X.iloc[cutoff_idx:]
        y_train_up, y_test_up = y_up[:train_end_idx], y_up[cutoff_idx:]
        y_train_down, y_test_down = y_down[:train_end_idx], y_down[cutoff_idx:]
    else:
        logger.info("🏭 MODO PRODUCCIÓN: Full History.")
        X_train, X_test = X, None
        y_train_up, y_test_up = y_up, None
        y_train_down, y_test_down = y_down, None

    targets_config = [("UP", y_train_up, y_test_up), ("DOWN", y_train_down, y_test_down)]
    
    # Obtenemos parámetros base y preparamos para modificación dinámica
    base_xgb_params = config.META_MODEL_PARAMS["XGB_PARAMS"].copy()

    for direction, y_train, y_test in targets_config:
        logger.info(f"🔥 Entrenando Meta-Modelo [{direction}]...")
        
        # --- CÁLCULO DE SCALE_POS_WEIGHT (BALANCEO DINÁMICO) ---
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        
        if n_pos > 0:
            scale_weight = n_neg / n_pos
            logger.info(f"   ⚖️ Clase desbalanceada detectada. Positivos: {n_pos} | Negativos: {n_neg}")
            logger.info(f"   ⚖️ Aplicando scale_pos_weight: {scale_weight:.2f}")
            base_xgb_params["scale_pos_weight"] = scale_weight
        else:
            logger.warning("   ⚠️ No hay casos positivos en el set de entrenamiento. El modelo será trivial.")
            base_xgb_params["scale_pos_weight"] = 1.0

        model = xgb.XGBClassifier(**base_xgb_params)
        
        eval_set = [(X_train, y_train)]
        if test_mode: eval_set.append((X_test, y_test))

        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        if test_mode:
            score = model.score(X_test, y_test)
            logger.info(f"   🏆 Accuracy [{direction}]: {score:.4f}")

        path = os.path.join(MODEL_DIR, f"xgboost_meta_{direction.lower()}.json")
        model.save_model(path)
        logger.info(f"   💾 Modelo guardado: {path}")

    elapsed = datetime.now() - start_time
    logger.info(f"✅ Pipeline Finalizado en {elapsed.total_seconds():.2f}s")

if __name__ == "__main__":
    train_meta_model()