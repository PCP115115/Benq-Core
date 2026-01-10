import polars as pl
import lightgbm as lgb
import joblib
import os
import sys

# Hack para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(engine_dir)

import config
import safety_mini_models as safety

def train_volatility_model(df_ticker: pl.DataFrame, ticker: str, save_dir: str):
    """
    Modelo de Volatilidad (Expansión).
    Target 1: Volatilidad YZ futura > Volatilidad YZ actual * 1.05
    """
    horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
    features_base = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
    features_rob = [f"{f}_rob" for f in features_base]
    
    # Usamos la volatilidad RAW para el target (comparación porcentual real)
    vol_col = f"vol_yz_{config.FEATURES_PARAMS['YANG_ZHANG_WINDOW']}d"
    
    # Target: ¿La vol dentro de 5 días es un 5% mayor que hoy?
    df_train = df_ticker.with_columns(
        (pl.col(vol_col).shift(-horizon) > (pl.col(vol_col) * 1.05))
        .cast(pl.Int8)
        .alias("target_vol")
    )
    
    df_train = df_train.drop_nulls(subset=features_rob + ["target_vol"])
    
    if df_train.height < 100: return None

    split_idx = int(df_train.height * 0.8)
    X_train = df_train[:split_idx].select(features_rob).to_pandas()
    y_train = df_train[:split_idx]["target_vol"].to_pandas()
    
    model = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model.fit(X_train, y_train)
    
    path = safety.get_model_path(save_dir, ticker, "volatility")
    joblib.dump(model, path)
    return path