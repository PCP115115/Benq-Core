import polars as pl
import lightgbm as lgb
import joblib
import os
import sys

# Hack para imports relativos
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(engine_dir)

import config
import safety_mini_models as safety

def train_trend_model(df_ticker: pl.DataFrame, ticker: str, save_dir: str):
    """
    Modelo de Tendencia (Triple Barrier modificada).
    Target 1: High toca Techo antes que Low toque Suelo (o horizonte final).
    """
    horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
    features_base = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
    # Usamos features robustas
    features_rob = [f"{f}_rob" for f in features_base] 
    
    # 1. Creación del Target (Triple Barrier)
    # Necesitamos mirar al futuro. Usamos shift negativo.
    
    # Expresiones para comprobar si High(t+k) > Ceil(t)
    ceil_col = f"fprice_ceil_yz_{horizon}d"
    floor_col = f"fprice_floor_yz_{horizon}d"
    
    # Lógica vectorizada:
    # Verificamos si en la ventana t+1 a t+H, el High supera el Techo actual
    target_exprs = []
    for i in range(1, horizon + 1):
        # Hit alcista: High futuro > Techo actual
        hit_up = (pl.col("High").shift(-i) >= pl.col(ceil_col))
        target_exprs.append(hit_up)
    
    # Target = 1 si ALGÚN día futuro rompe el techo
    df_train = df_ticker.with_columns(
        pl.any_horizontal(target_exprs).cast(pl.Int8).alias("target_trend")
    )
    
    # Limpieza de nulos generados por shift y validación
    df_train = df_train.drop_nulls(subset=features_rob + ["target_trend"])
    
    if df_train.height < 100:
        return None # Datos insuficientes

    # 2. Train/Test Split (Temporal)
    split_idx = int(df_train.height * 0.8)
    X_train = df_train[:split_idx].select(features_rob).to_pandas()
    y_train = df_train[:split_idx]["target_trend"].to_pandas()
    
    # 3. Entrenamiento
    model = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model.fit(X_train, y_train)
    
    # 4. Guardado
    path = safety.get_model_path(save_dir, ticker, "trend")
    joblib.dump(model, path)
    return path