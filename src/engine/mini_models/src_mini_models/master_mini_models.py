import sys
import os
import logging
import concurrent.futures
import polars as pl
from typing import List

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(engine_dir)
sys.path.append(os.path.join(engine_dir, "src_features"))

import config
from src_features import master_features
import safety_mini_models as safety
import trend_mini_model
import volatility_mini_models
import reversion_mini_models

# Configuración Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MINI-MODELS] - %(levelname)s - %(message)s')
logger = logging.getLogger("MasterMiniModels")

def _worker_train_ticker(ticker: str, data_path: str, models_dir: str):
    """
    Función Worker para ProcessPoolExecutor.
    Recibe el ticker y la ruta del parquet (evita serializar todo el DF).
    """
    try:
        # 1. Carga de datos optimizada (Lazy Scan + Filter)
        # Usamos layer="robust" como se solicitó
        # Nota: master_features.get_feature_matrix no es pickleable fácilmente si tiene locks,
        # mejor leer el parquet directamente aquí o filtrar de un DF pasado.
        # Por seguridad y constraints de 'master_features', usamos la API oficial.
        
        # Filtrado in-memory (Trade-off: memoria vs I/O)
        # Asumimos que data_path apunta al parquet generado por master_features
        df_ticker = pl.read_parquet(data_path).filter(pl.col("ticker") == ticker)
        
        if df_ticker.is_empty():
            return f"{ticker}: SKIPPED (No data)"

        res_msgs = []
        
        # 2. Entrenar Modelo de Tendencia
        try:
            p_trend = trend_mini_model.train_trend_model(df_ticker, ticker, models_dir)
            if p_trend: res_msgs.append("Trend OK")
        except Exception as e:
            res_msgs.append(f"Trend ERR: {e}")

        # 3. Entrenar Modelo de Volatilidad
        try:
            p_vol = volatility_mini_models.train_volatility_model(df_ticker, ticker, models_dir)
            if p_vol: res_msgs.append("Vol OK")
        except Exception as e:
            res_msgs.append(f"Vol ERR: {e}")
            
        # 4. Entrenar Modelo de Reversión
        try:
            p_rev = reversion_mini_models.train_reversion_model(df_ticker, ticker, models_dir)
            if p_rev: res_msgs.append("Rev OK")
        except Exception as e:
            res_msgs.append(f"Rev ERR: {e}")

        return f"{ticker}: " + " | ".join(res_msgs)

    except Exception as e:
        return f"{ticker}: CRITICAL FAIL ({e})"

def train_all_models(tickers: List[str] = None):
    """
    Orquestador principal. Entrena mini-modelos en paralelo.
    """
    logger.info("🚀 Iniciando entrenamiento masivo de Mini-Modelos...")
    
    models_dir = os.path.join(config.PATHS["MINI_MODELS_DIR"])
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Asegurar que tenemos features actualizadas
    logger.info("📡 Obteniendo Feature Matrix (Robust Layer)...")
    # Llamamos a master para asegurar actualización (Cold Start check)
    # No filtramos por ticker aquí para aprovechar el caché global del pipeline
    full_df = master_features.get_feature_matrix(layer="robust")
    
    if full_df.is_empty():
        logger.error("❌ No hay datos disponibles para entrenar.")
        return

    # Guardamos temporalmente si no existe persistencia, o usamos la ruta configurada
    data_path = os.path.join(os.path.dirname(engine_dir), config.PATHS["FEATURES_OUTPUT"])
    
    if not tickers:
        tickers = full_df["ticker"].unique().to_list()
    
    logger.info(f"⚙️ Entrenando expertos para {len(tickers)} activos usando Multiprocessing...")
    
    # 2. Ejecución Paralela
    # Usamos ProcessPoolExecutor porque el entrenamiento de LGBM libera el GIL, 
    # pero la preparación de datos en Polars/Python compite por CPU.
    max_workers = os.cpu_count() - 1 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Mapeamos tarea a futuro
        future_to_ticker = {
            executor.submit(_worker_train_ticker, t, data_path, models_dir): t 
            for t in tickers
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            tk = future_to_ticker[future]
            try:
                res = future.result()
                logger.info(f"   {res}")
            except Exception as e:
                logger.error(f"❌ Error en proceso worker para {tk}: {e}")

    logger.info("✅ Entrenamiento masivo finalizado.")

if __name__ == "__main__":
    # Test manual
    train_all_models(tickers=["AAPL", "MSFT"])