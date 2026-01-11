import polars as pl
import lightgbm as lgb
import joblib
import numpy as np
import sys
from pathlib import Path

#manejo de rutas:
current_file = Path(__file__).resolve()
src_path = current_file.parents[3] 
features_path = src_path / "engine" / "src_features" 
sys.path.append(str(src_path))
sys.path.append(str(features_path)) 

from engine import config
from engine.src_features import master_features, indicators
from engine.mini_models.src_mini_models import safety_mini_models as safety



def train_trend_model(ticker: str, save_dir: str):
    # --- 1. CARGA Y PREPARACIÓN ---
    
    horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
    z_score = config.FEATURES_PARAMS["YZ_Z_SCORE"]
    features_trend = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
    layer = config.MINI_MODEL_PARAMS["LAYER"][0]
    
    print(f"--- Procesando {ticker} [Dual Breakout] ---")
    
    # Cargamos datos (Raw para precios, Rob para features)
    df = master_features.get_feature_matrix(tickers=ticker, layer = layer, features=None)
    print(df.columns)
    
    if df is None or df.height < 200:
        print(f"Skipping {ticker}: Datos insuficientes.")
        return None

    # --- 2. GENERAR BARRERAS (TECHO Y SUELO) ---
    vol_col = f"vol_yz_{config.FEATURES_PARAMS['YANG_ZHANG_WINDOW']}d"
    
    bounds_exprs = indicators.get_volatility_bounds(
        col_close="Close",
        col_vol_yz=vol_col,
        z_score=z_score,
        horizon=horizon
    )
    df = df.with_columns(bounds_exprs)
    
    ceil_col = f"fprice_ceil_yz_{horizon}d"
    floor_col = f"fprice_floor_yz_{horizon}d"

    # --- 3. LÓGICA DE CARRERA (RACE LOGIC) ---
    # Calculamos qué ocurre primero
    days_to_ceil = []
    days_to_floor = []
    
    for i in range(1, horizon + 1):
        future_high = pl.col("High").shift(-i)
        future_low = pl.col("Low").shift(-i)
        
        # Si toca techo/suelo en el día 'i', guardamos 'i', si no 999
        hit_c = pl.when(future_high > pl.col(ceil_col)).then(i).otherwise(999)
        hit_f = pl.when(future_low < pl.col(floor_col)).then(i).otherwise(999)
        
        days_to_ceil.append(hit_c)
        days_to_floor.append(hit_f)
    
    # Encontramos el día del primer evento
    df = df.with_columns([
        pl.min_horizontal(days_to_ceil).alias("first_ceil_hit"),
        pl.min_horizontal(days_to_floor).alias("first_floor_hit")
    ])

    # --- 4. DEFINICIÓN DE LOS DOS TARGETS ---
    
    # Target UP: Toca techo (hit != 999) Y ocurre ANTES que el suelo
    target_up_expr = (
        (pl.col("first_ceil_hit") != 999) & 
        (pl.col("first_ceil_hit") < pl.col("first_floor_hit"))
    ).cast(pl.Int8)

    # Target DOWN: Toca suelo (hit != 999) Y ocurre ANTES que el techo
    target_down_expr = (
        (pl.col("first_floor_hit") != 999) & 
        (pl.col("first_floor_hit") < pl.col("first_ceil_hit"))
    ).cast(pl.Int8)
    
    df_train = df.with_columns([
        target_up_expr.alias("target_up"),
        target_down_expr.alias("target_down")
    ])

    # --- 5. LIMPIEZA COMÚN ---
    features_cols = features_trend 
    print(f"Número de features seleccionadas: {len(features_cols)}")
    print("Primeros 10 features:")
    for feature in features_cols[:10]:
        print(f"  - {feature}")

    missing_cols = [c for c in features_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ ERROR CRÍTICO: Las siguientes columnas no están en el DataFrame: {missing_cols}")
        return None
    
    cols_to_clean = features_cols + ["target_up", "target_down"]
    df_train = df_train.drop_nulls(subset=cols_to_clean)
    
    if df_train.height < 100: 
        print(f"Skipping {ticker}: Pocos datos tras limpieza ({df_train.height} filas).")
        return None
    
    # =======================================================
    # BLOQUE DE SPLIT CONFIGURABLE (TEST vs PRODUCCIÓN)
    # =======================================================
    
    # Cargamos parámetros del config actualizado
    train_params = config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]
    
    split_ratio = train_params["TRAIN_TEST_SPLIT_RATIO"]
    is_test_mode = train_params["TEST_MODE"]
    use_purge = train_params["PURGE_OVERLAP"]

    # Variables para almacenar los datasets finales
    X_train_pd = None
    y_train_up = None
    y_train_down = None

    if not is_test_mode:
        # --- MODO PRODUCCIÓN: 100% DATOS ---
        print("🚀 MODO PRODUCCIÓN: Entrenando con el 100% de los datos.")
        
        X_train_pd = df_train.select(features_cols).to_pandas()
        y_train_up = df_train["target_up"].to_pandas()
        y_train_down = df_train["target_down"].to_pandas()
        
    else:
        # --- MODO TEST: SPLIT + PURGE ---
        print(f"🧪 MODO TEST: Split {split_ratio:.0%} | Purge activado: {use_purge}")
        
        cutoff_index = int(df_train.height * split_ratio)
        purge_size = horizon if use_purge else 0
        train_end_idx = cutoff_index - purge_size
        
        if train_end_idx < 100:
             print(f"Skipping {ticker}: Pocos datos para el Split configurado.")
             return None

        # Generación de sets con slicing (Train set purgado)
        X_train_pd = df_train[:train_end_idx].select(features_cols).to_pandas()
        y_train_up = df_train[:train_end_idx]["target_up"].to_pandas()
        y_train_down = df_train[:train_end_idx]["target_down"].to_pandas()
        
        # Logging informativo del split
        print(f"   Train: 0 -> {train_end_idx} | Gap: {purge_size} | Test: {cutoff_index} -> End")

    # Inicializamos el diccionario de retorno (Renombrado para evitar warning)
    model_paths = {}

    # --- 6. ENTRENAMIENTO MODELO UP (CALL) ---
    print(f"  > Entrenando modelo UP (Long)...")
    model_up = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model_up.fit(X_train_pd, y_train_up)
    
    path_up = safety.get_model_path(save_dir, ticker, "yz_breakout_up")
    joblib.dump(model_up, path_up)
    model_paths["up"] = path_up

    # --- 7. ENTRENAMIENTO MODELO DOWN (PUT) ---
    print(f"  > Entrenando modelo DOWN (Short)...")
    model_down = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model_down.fit(X_train_pd, y_train_down)
    
    path_down = safety.get_model_path(save_dir, ticker, "yz_breakout_down")
    joblib.dump(model_down, path_down)
    model_paths["down"] = path_down
    
    return model_paths


