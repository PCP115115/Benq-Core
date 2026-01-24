import polars as pl
import joblib
import pandas as pd
import numpy as np
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 0. SETUP DE RUTAS ---
current_file = Path(__file__).resolve()
src_path = current_file.parents[3]
features_path = src_path / "engine" / "src_features"
sys.path.append(str(src_path))
sys.path.append(str(features_path))

from engine import config
from engine.src_features import master_features
# Importamos los 3 Expertos
from engine.mini_models.src_mini_models import (
    trend_mini_model,
    reversion_mini_models,
    volatility_mini_models
)

# --- RUTAS DE SALIDA ---
# Aquí se guardarán las tablas para el Meta-Modelo
META_INPUT_DIR = src_path / "data" / "processed" / "meta_model_inputs"
MODELS_DIR = src_path / "data" / "models" / "mini_models"

def get_test_data(df: pl.DataFrame):
    """
    Replica EXACTAMENTE la lógica de Split de los scripts de entrenamiento
    para obtener el conjunto de TEST limpio y virgen.
    """
    train_params = config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]
    split_ratio = train_params["TRAIN_TEST_SPLIT_RATIO"]
    
    # El punto de corte es el mismo que usan los modelos para dejar de entrenar
    cutoff_index = int(df.height * split_ratio)
    
    # Seleccionamos desde el cutoff hasta el final (TEST SET)
    df_test = df[cutoff_index:]
    return df_test

def process_ticker_pipeline(ticker: str):
    """
    WORKER FUNCTION: Se ejecuta en un núcleo separado del procesador.
    1. Carga datos una vez.
    2. Entrena los 3 modelos (Trend, Reversion, Volatility).
    3. Carga los modelos entrenados.
    4. Genera predicciones (Probabilidades) sobre el Test Set.
    5. Guarda el archivo input para el Meta-Modelo.
    """
    try:
        print(f"💎 [{ticker}] Iniciando Pipeline de Expertos...")
        
        # 1. Carga de Datos (Common Layer)
        layer = config.MINI_MODEL_PARAMS["LAYER"][0]
        # features=None para traer TODO el contexto
        df = master_features.get_feature_matrix(tickers=ticker, layer=layer, features=None)
        
        if df is None or df.height < 200:
            return f"⚠️ {ticker}: Datos insuficientes."

        # -------------------------------------------------------------
        # NOTA IMPORTANTE PARA VERSIÓN 2.0 (FUTURO):
        # ACTUALMENTE USAMOS UN SIMPLE SPLIT (TRAIN/TEST).
        # EN PRODUCCIÓN REAL DEBERÍAMOS USAR 'PURGED K-FOLD CROSS VALIDATION'
        # PARA GENERAR PROBABILIDADES "OUT-OF-SAMPLE" PARA TODO EL HISTÓRICO.
        # PERO PARA LA V1.0, ESTO ES SUFICIENTE Y ROBUSTO.
        # -------------------------------------------------------------

        # 2. Obtenemos el Test Set (Donde haremos las predicciones)
        # Esto asegura que las probabilidades que ve el Meta-Modelo son "honestas"
        df_test = get_test_data(df)
        
        if df_test.height < 10:
            return f"⚠️ {ticker}: Test set demasiado pequeño."

        # DataFrame base para el Meta-Modelo (Date, Close, Returns)
        # Convertimos a Pandas porque los modelos de ML esperan Pandas/Numpy
        meta_df = df_test.select(["Date", "ticker", "Close", "log_returns"]).to_pandas()
        
        # Directorio de modelos para este ticker
        save_dir = str(MODELS_DIR)

        # ==============================================================================
        # FASE A: ENTRENAMIENTO & INFERENCIA - EXPERTO 1: TREND
        # ==============================================================================
        # 1. Entrenar
        trend_paths = trend_mini_model.train_trend_model(ticker, save_dir)
        
        if trend_paths:
            # 2. Cargar Modelos
            model_up = joblib.load(trend_paths["up"])
            model_down = joblib.load(trend_paths["down"])
            
            # 3. Preparar Features (Solo las que necesita Trend)
            feats = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
            X_test = df_test.select(feats).to_pandas()
            
            # 4. Predecir Probabilidades (Columna 1 = Probabilidad de clase 1)
            meta_df["P_Trend_Up"] = model_up.predict_proba(X_test)[:, 1]
            meta_df["P_Trend_Down"] = model_down.predict_proba(X_test)[:, 1]
        else:
            meta_df["P_Trend_Up"] = 0.5 # Valor neutro por defecto si falla
            meta_df["P_Trend_Down"] = 0.5

        # ==============================================================================
        # FASE B: ENTRENAMIENTO & INFERENCIA - EXPERTO 2: REVERSION
        # ==============================================================================
        rev_paths = reversion_mini_models.train_reversion_model(ticker, save_dir)
        
        if rev_paths:
            model_up = joblib.load(rev_paths["up"])
            model_down = joblib.load(rev_paths["down"])
            feats = config.MINI_MODEL_PARAMS["FEATURES_REVERSION"]
            X_test = df_test.select(feats).to_pandas()
            
            meta_df["P_Rev_Up"] = model_up.predict_proba(X_test)[:, 1]
            meta_df["P_Rev_Down"] = model_down.predict_proba(X_test)[:, 1]
        else:
            meta_df["P_Rev_Up"] = 0.5
            meta_df["P_Rev_Down"] = 0.5

        # ==============================================================================
        # FASE C: ENTRENAMIENTO & INFERENCIA - EXPERTO 3: VOLATILITY
        # ==============================================================================
        vol_paths = volatility_mini_models.train_volatility_model(ticker, save_dir)
        
        if vol_paths:
            model_exp = joblib.load(vol_paths["expansion"])
            model_com = joblib.load(vol_paths["compression"])
            feats = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
            X_test = df_test.select(feats).to_pandas()
            
            meta_df["P_Vol_Exp"] = model_exp.predict_proba(X_test)[:, 1]
            meta_df["P_Vol_Comp"] = model_com.predict_proba(X_test)[:, 1]
        else:
            meta_df["P_Vol_Exp"] = 0.5
            meta_df["P_Vol_Comp"] = 0.5

        # ==============================================================================
        # FASE D: GUARDADO FINAL
        # ==============================================================================
        # Convertimos de vuelta a Polars para guardar eficiente en Parquet
        final_pl = pl.from_pandas(meta_df)
        
        output_path = META_INPUT_DIR / f"meta_input_{ticker}.parquet"
        os.makedirs(META_INPUT_DIR, exist_ok=True)
        final_pl.write_parquet(output_path)
        
        return f"✅ {ticker}: Completado. Output en {output_path.name}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ {ticker}: Error crítico - {str(e)}"

def run_mini_models_pipeline(tickers: list, n_jobs: int = -1):
    """
    Orquestador Principal.
    Args:
        tickers: Lista de símbolos a procesar.
        n_jobs: Número de núcleos (-1 = Todos los disponibles).
    """
    start_time = time.time()
    
    # Si n_jobs es -1, usar cpu_count - 1 para dejar uno libre al sistema
    if n_jobs == -1:
        n_jobs = max(1, os.cpu_count() - 1)
    
    print("\n" + "="*60)
    print(f"🚀 INICIANDO MASTER PIPELINE DE MINI-MODELOS")
    print(f"🎯 Tickers: {len(tickers)}")
    print(f"🧠 Workers: {n_jobs}")
    print("="*60 + "\n")

    results = []
    
    # Ejecución Paralela
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Lanzamos tareas
        future_to_ticker = {executor.submit(process_ticker_pipeline, t): t for t in tickers}
        
        # Recogemos resultados conforme acaban
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as exc:
                print(f"❌ {ticker} generó una excepción: {exc}")

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"🏁 PIPELINE FINALIZADO en {elapsed:.2f} segundos.")
    print("="*60)


if __name__ == "__main__":
    try:
        # Intentamos importar la lista de tickers de la ESTRATEGIA
        # Esto asegura que procesamos los activos correctos (los 21 nuevos)
        from src.strategy.config_strategy import TICKERS_ESTRATEGIA
        print("📋 Usando lista de tickers de ESTRATEGIA (config_strategy.py)")
        target_tickers = TICKERS_ESTRATEGIA
    except ImportError:
        # Fallback si no se encuentra (para pruebas aisladas)
        print("⚠️ No se encontró config_strategy. Usando lista de prueba simple.")
        target_tickers = ["AAPL", "MSFT", "GOOGL"]

    # Ejecutar el pipeline
    run_mini_models_pipeline(target_tickers, n_jobs=15)
