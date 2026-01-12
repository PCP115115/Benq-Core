import polars as pl
import lightgbm as lgb
import joblib
import numpy as np
import sys
import os
from pathlib import Path

# --- 0. SETUP DE RUTAS ROBUSTO ---
# Permite ejecución desde cualquier nivel del proyecto
current_file = Path(__file__).resolve()
src_path = current_file.parents[3] 
features_path = src_path / "engine" / "src_features" 
sys.path.append(str(src_path))
sys.path.append(str(features_path)) 

from engine import config
from engine.src_features import master_features
from engine.mini_models.src_mini_models import safety_mini_models as safety

def train_volatility_model(ticker: str, save_dir: str):
    """
    Entrena DOS modelos de Régimen de Volatilidad (Expertos en Magnitud):
    
    1. 'vol_expansion':   Probabilidad de que la volatilidad AUMENTE (Mercado Nervioso/Tormenta).
    2. 'vol_compression': Probabilidad de que la volatilidad DISMINUYA (Mercado en Calma/Rango).
    
    BLINDAJE:
    - Target: Calculated on Future Realized Volatility vs Current Implied/Hist Volatility.
    - Split:  Purged Time-Series Split para evitar Look-Ahead Bias.
    """
    
    # --- 1. CARGA DE CONFIGURACIÓN Y PREPARACIÓN ---
    try:
        horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
        features_vol = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
        layer = config.MINI_MODEL_PARAMS["LAYER"][0]
        # Parámetro extra de seguridad para evitar divisiones por cero o ruido en targets
        train_params = config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]
    except KeyError as e:
        print(f"❌ ERROR CONFIG: Falta el parámetro {e} en config.py")
        return None
    
    # Seguridad: Crear directorio de salida si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"--- Procesando {ticker} [Volatility Regime Expert] ---")
    
    # --- 2. CARGA DE DATOS (CONTEXTO COMPLETO) ---
    # Usamos features=None para traer log_returns y precios necesarios para el target
    df = master_features.get_feature_matrix(tickers=ticker, layer=layer, features=None)
    
    # Blindaje: Verificación de datos mínimos
    if df is None:
        print(f"Skipping {ticker}: DataFrame es None.")
        return None
        
    if df.height < 200:
        print(f"Skipping {ticker}: Datos insuficientes ({df.height} filas).")
        return None

    # Blindaje: Verificación de columna crítica para target
    if "log_returns" not in df.columns:
        print(f"❌ ERROR CRÍTICO: 'log_returns' no encontrado. Necesario para calcular volatilidad futura.")
        return None

    # --- 3. INGENIERÍA DEL TARGET (LA LÓGICA DE LA TORMENTA) ---
    # Definimos matemáticamente qué significa "El mercado se va a poner nervioso"
    
    # A. Volatilidad Actual (Baseline):
    # Usamos la mejor estimación disponible hoy. En tu config es 'vol_yz_20d'.
    # Si por alguna razón falla, usamos la desviación estándar simple.
    if "vol_yz_20d" in df.columns:
        current_vol = pl.col("vol_yz_20d")
    else:
        current_vol = pl.col("vol_std_20d")
    
    # B. Volatilidad Futura (Realized Volatility):
    # ¿Cuánto se movió REALMENTE el precio en los siguientes 'horizon' días?
    # Usamos rolling_std de los retornos logarítmicos mirando hacia adelante (shift negativo).
    future_vol = (
        pl.col("log_returns")
        .rolling_std(window_size=horizon)
        .shift(-horizon) # <--- EL SECRETO: Alineamos el futuro con la fila actual
    )

    # C. Definición de Targets Binarios
    # EXPANSION: Si la volatilidad futura es mayor que la actual.
    target_exp_expr = (future_vol > current_vol).cast(pl.Int8)
    
    # COMPRESSION: Si la volatilidad futura es menor o igual.
    target_com_expr = (future_vol <= current_vol).cast(pl.Int8)
    
    df = df.with_columns([
        target_exp_expr.alias("target_expansion"),
        target_com_expr.alias("target_compression")
    ])

    # --- 4. LIMPIEZA Y SELECCIÓN DE FEATURES ---
    print(f"Número de features seleccionadas: {len(features_vol)}")
    print("Primeros features de Volatilidad:")
    for feature in features_vol[:5]:
        print(f"  - {feature}")

    # Blindaje: Asegurar que todas las features existen
    missing_cols = [c for c in features_vol if c not in df.columns]
    if missing_cols:
        print(f"❌ ERROR CRÍTICO: Faltan columnas configuradas: {missing_cols}")
        return None
    
    # Limpieza estricta: Borramos filas donde falten features O targets
    # (Los targets serán nulos en las últimas 'horizon' filas por el shift)
    cols_to_clean = features_vol + ["target_expansion", "target_compression"]
    df_train = df.drop_nulls(subset=cols_to_clean)
    
    if df_train.height < 100: 
        print(f"Skipping {ticker}: Pocos datos tras limpieza ({df_train.height} filas).")
        return None

    # =======================================================
    # BLOQUE DE SPLIT ROBUSTO (PURGED TIME-SERIES SPLIT)
    # =======================================================
    
    split_ratio = train_params["TRAIN_TEST_SPLIT_RATIO"]
    is_test_mode = train_params["TEST_MODE"]
    use_purge = train_params["PURGE_OVERLAP"]

    X_train_pd = None
    y_train_exp = None
    y_train_com = None

    if not is_test_mode:
        # --- MODO PRODUCCIÓN (Live Trading) ---
        print("🚀 MODO PRODUCCIÓN: Entrenando con el 100% de los datos históricos.")
        # En producción usamos todo para que el modelo conozca el régimen actual
        X_train_pd = df_train.select(features_vol).to_pandas()
        y_train_exp = df_train["target_expansion"].to_pandas()
        y_train_com = df_train["target_compression"].to_pandas()
        
    else:
        # --- MODO TEST (Backtesting Seguro) ---
        print(f"🧪 MODO TEST: Split {split_ratio:.0%} | Purge activado: {use_purge}")
        
        cutoff_index = int(df_train.height * split_ratio)
        
        # BLINDAJE ANTI-LEAK: Borramos los datos que se solapan con el futuro del test
        purge_size = horizon if use_purge else 0
        train_end_idx = cutoff_index - purge_size
        
        if train_end_idx < 100:
             print(f"Skipping {ticker}: Datos insuficientes para realizar Split con Purge.")
             return None

        # Slicing Seguro
        X_train_pd = df_train[:train_end_idx].select(features_vol).to_pandas()
        y_train_exp = df_train[:train_end_idx]["target_expansion"].to_pandas()
        y_train_com = df_train[:train_end_idx]["target_compression"].to_pandas()
        
        print(f"   Train: 0 -> {train_end_idx} | Gap (Purga): {purge_size} | Test Start: {cutoff_index}")

    model_paths = {}

    # --- 5. ENTRENAMIENTO MODELO EXPANSION (LONG VOL) ---
    print(f"  > Entrenando modelo EXPANSION (Risk ON)...")
    model_exp = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model_exp.fit(X_train_pd, y_train_exp)
    
    path_exp = safety.get_model_path(save_dir, ticker, "vol_expansion")
    joblib.dump(model_exp, path_exp)
    model_paths["expansion"] = path_exp

    # --- 6. ENTRENAMIENTO MODELO COMPRESSION (SHORT VOL) ---
    print(f"  > Entrenando modelo COMPRESSION (Risk OFF)...")
    model_com = lgb.LGBMClassifier(**config.MINI_MODEL_PARAMS["LGBM_PARAMS"])
    model_com.fit(X_train_pd, y_train_com)
    
    path_com = safety.get_model_path(save_dir, ticker, "vol_compression")
    joblib.dump(model_com, path_com)
    model_paths["compression"] = path_com
    
    return model_paths

