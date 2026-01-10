import polars as pl
import lightgbm as lgb
import joblib
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(engine_dir)

import config
import safety_mini_models as safety

def train_reversion_model(df_ticker: pl.DataFrame, ticker: str, save_dir: str):
    """
    Modelo de Reversión a la Media.
    Target 1: Precio cruza SMA_15 en el horizonte futuro.
    """
    horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
    features_base = config.MINI_MODEL_PARAMS["FEATURES_REVERSION"]
    features_rob = [f"{f}_rob" for f in features_base]
    
    # Target Logic: ¿Cruza el precio la SMA?
    # Significa que Low < SMA < High en algún momento futuro
    
    # Calculamos SMA_15 raw al vuelo si no viniera, pero en indicators.py 
    # tenemos 'rel_sma_15'. Necesitamos SMA absoluta para comparar precios.
    # Recalculamos SMA absoluta para el target.
    sma_col = pl.col("Close").rolling_mean(window_size=15)
    
    # Miramos al futuro
    target_exprs = []
    for i in range(1, horizon + 1):
        # Cross check: Low(t+i) < SMA(t+i) < High(t+i)
        # Nota: Usamos SMA shifted también porque comparamos en el tiempo t+i
        future_sma = sma_col.shift(-i)
        future_high = pl.col("High").shift(-i)
        future_low = pl.col("Low").shift(-i)
        
        cross = (future_low < future_sma) & (future_high > future_sma)
        target_exprs.append(cross)
        
    df_train = df_ticker.with_columns(
        pl.any_horizontal(target_exprs).cast(pl.Int8).alias("target_reversion")
    )
    
    df_train = df_train.drop_nulls(subset=features_rob + ["target_reversion"])
    
    if df_train.height < 100: return None

    split_idx = int(df_train.height * 0.8)
    X_train = df_train[:split_idx].select(features_rob).to_pandas()
    y_train = df_train[:split_idx]["target_reversion"].to_pandas()
    
    model = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model.fit(X_train, y_train)
    
    path = safety.get_model_path(save_dir, ticker, "reversion")
    joblib.dump(model, path)
    return path