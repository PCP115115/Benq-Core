import sys
import os
import polars as pl
from pathlib import Path

# --- SETUP DE RUTAS ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.engine.mini_models.src_mini_models.master_mini_models import run_mini_models_pipeline
from src.engine.config import META_MODEL_PARAMS, FEATURES_PARAMS
import src.engine.config as config
from src.engine.context.master_context import get_market_regime
from src.engine.src_features.master_features import get_feature_matrix

# Variables globales
START_DATE = META_MODEL_PARAMS["start_date"]
END_DATE = META_MODEL_PARAMS["end_date"]
FEATURE_LIST = META_MODEL_PARAMS["feature_list"]
NORM_WINDOW = META_MODEL_PARAMS["normalization_window"]
TICKER = config.ticker_analizado 
LAYER = config.layer

def get_data_meta_model(ticker, start_date, end_date, layer, feature_list, normalization_window):
    print(f"🧩 Construyendo Dataset Meta-Modelo para {ticker}...")

    # 1. FEATURES NORMALIZADOS (INPUT DEL MODELO - X)
    print("   > Obteniendo Features (Normalizados)...")
    df_features_norm = get_feature_matrix(
        tickers=ticker, 
        start_date=start_date, 
        end_date=end_date, 
        layer=layer, 
        features=feature_list, 
        normalization_window=normalization_window
    )
    
    # 2. DATOS CRUDOS (PARA CÁLCULO DE TARGET - Y)
    # Necesitamos precios y la volatilidad CRUDA para calcular las barreras correctamente.
    # La volatilidad normalizada (Z-Score) NO sirve para proyectar precios.
    print("   > Obteniendo Datos Crudos (Precios + Volatilidad)...")
    
    vol_window = FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
    vol_col_name = f"vol_yz_{vol_window}d"
    
    # Pedimos explícitamente las columnas raw
    raw_columns = ["Close", "High", "Low", "Open", vol_col_name]
    
    df_raw = get_feature_matrix(
        tickers=ticker,
        start_date=start_date,
        end_date=end_date,
        layer="all",       # Layer all para acceder a todo
        features=raw_columns,
        normalization_window=None # IMPORTANTE: None para obtener valores reales
    )
    
    # Renombramos la volatilidad cruda para evitar colisión con la normalizada
    # y seleccionamos solo lo necesario para no duplicar Date/Ticker erróneamente en el join
    df_raw = df_raw.select(["Date", "ticker"] + raw_columns).rename({
        vol_col_name: f"{vol_col_name}_RAW"
    })

    # 3. CONTEXTO DE MERCADO
    print("   > Obteniendo Contexto de Mercado...")
    df_context = get_market_regime(tickers=[ticker])
    if not df_context.is_empty():
        df_context = df_context.select(["Date", "ticker", "market_regime", "regime_probability"])

    # 4. MINI MODELS (EXPERTOS)
    print("   > Ejecutando Pipeline de Expertos (Mini-Models)...")
    run_mini_models_pipeline(tickers=[ticker], n_jobs=-1)
    
    path_mini_models = os.path.join(
        project_root, "src", "data", "processed", "meta_model_inputs", f"meta_input_{ticker}.parquet"
    )
    
    if not os.path.exists(path_mini_models):
        raise FileNotFoundError(f"❌ Error: No se generó el output de mini-modelos en {path_mini_models}")
    
    df_mini_models = pl.read_parquet(path_mini_models)
    
    # Seleccionamos solo las probabilidades
    cols_expertos = [c for c in df_mini_models.columns if c.startswith("P_") or c in ["Date", "ticker"]]
    df_mini_models = df_mini_models.select(cols_expertos)

    # 5. UNIFICACIÓN FINAL
    print("   > Unificando DataFrames...")
    
    # Join Progresivo
    # Base: Features Normalizados
    meta_model_data = df_features_norm
    
    # + Raw Data (Inner Join)
    meta_model_data = meta_model_data.join(df_raw, on=["Date", "ticker"], how="inner")
    
    # + Contexto (Inner Join)
    if not df_context.is_empty():
        meta_model_data = meta_model_data.join(df_context, on=["Date", "ticker"], how="inner")
        
    # + Expertos (Inner Join - Esto recorta al periodo de test si TEST_MODE=True)
    meta_model_data = meta_model_data.join(df_mini_models, on=["Date", "ticker"], how="inner")
    
    meta_model_data = meta_model_data.sort("Date")
    
    print(f"✅ Dataset Completo Generado: {meta_model_data.height} filas.")
    return meta_model_data

if __name__ == "__main__":
    try:
        df = get_data_meta_model(TICKER, START_DATE, END_DATE, LAYER, FEATURE_LIST, NORM_WINDOW)
        print(df.head())
        print("Columnas:", df.columns)
    except Exception as e:
        print(f"Error: {e}")