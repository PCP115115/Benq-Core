import sys
import os
import polars as pl
import torch
import numpy as np # Necesario para split vectorial
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(engine_dir)
project_root = os.path.dirname(src_dir)

sys.path.append(src_dir)
sys.path.append(engine_dir)

try:
    import config
    from src_features import master_features
    from auto_encoder_lstm import LSTMHandler
    from gmm_model import RegimeDetector
except ImportError as e:
    print(f"❌ Error de importación en Context: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CONTEXT] - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketContext")

def get_market_regime(tickers=None, date=None) -> pl.DataFrame:
    """
    Función maestra optimizada con INFERENCIA VECTORIZADA.
    """
    params = config.CONTEXT_PARAMS
    path_lstm = os.path.join(project_root, config.PATHS["MODEL_LSTM"])
    path_gmm = os.path.join(project_root, config.PATHS["MODEL_GMM"])
    
    # 1. OBTENCIÓN DE DATOS
    logger.info("📡 Solicitando datos normalizados al vuelo...")
    df_input = master_features.get_feature_matrix(
        tickers=tickers, 
        features=params["INPUT_FEATURES"],
        normalization_window=params["NORMALIZATION_WINDOW"]
    )
    
    if df_input.is_empty():
        logger.error("No hay datos disponibles para Context.")
        return pl.DataFrame()

    feat_cols = params["INPUT_FEATURES"]
    input_dim = len(feat_cols)
    lstm_handler = LSTMHandler(input_dim, params)
    gmm_model = RegimeDetector(params["GMM_N_COMPONENTS"])

    models_exist = os.path.exists(path_lstm) and os.path.exists(path_gmm)

    # ---------------------------------------------------------
    # FASE A: PREPARACIÓN DE DATOS (COMÚN PARA ENTRENAR E INFERIR)
    # ---------------------------------------------------------
    # Agrupamos por ticker para respetar la secuencia temporal
    unique_tickers = df_input["ticker"].unique().to_list()
    if isinstance(tickers, str): unique_tickers = [tickers] # Si pidió uno solo
    elif tickers: unique_tickers = [t for t in unique_tickers if t in tickers]

    # Contenedores para procesamiento por lotes
    tensor_batches = []
    meta_info = [] # Guardaremos (Ticker, Date_Series) para reconstruir luego
    
    logger.info(f"⚙️ Procesando estructuras de tensores para {len(unique_tickers)} tickers...")
    
    for tk in unique_tickers:
        # Ordenar es vital para LSTM
        df_tk = df_input.filter(pl.col("ticker") == tk).sort("Date")
        
        if df_tk.height <= params["LSTM_WINDOW_SIZE"]:
            continue
            
        try:
            # t_tk shape: [Samples, Window, Feats]
            t_tk, _ = lstm_handler.prepare_data(df_tk, feat_cols)
            tensor_batches.append(t_tk)
            
            # Guardamos la metadata alineada con el output del modelo
            # El modelo emite output para el índice [Window-1] en adelante
            # Ejemplo: Window=20. Primera predicción es para la fila 19 (día 20).
            valid_dates = df_tk["Date"].slice(params["LSTM_WINDOW_SIZE"] - 1, len(df_tk))
            meta_info.append((tk, valid_dates))
            
        except ValueError:
            continue

    if not tensor_batches:
        logger.warning("⚠️ Datos insuficientes para generar secuencias válidas.")
        return pl.DataFrame()

    # MEGA-TENSOR: Concatenamos todo el mercado en un solo bloque de memoria
    # Esto permite inferencia paralela masiva.
    full_tensor = torch.cat(tensor_batches, dim=0)
    logger.info(f"📦 Batch compilado: {full_tensor.shape[0]} ventanas totales.")

    # ---------------------------------------------------------
    # FASE B: AUTO-HEALING (ENTRENAMIENTO)
    # ---------------------------------------------------------
    if not models_exist:
        logger.warning("⚠️ Modelos no encontrados. Iniciando ENTRENAMIENTO completo...")
        
        # 1. Entrenar LSTM
        dataset = TensorDataset(full_tensor)
        loader = DataLoader(dataset, batch_size=params["LSTM_BATCH_SIZE"], shuffle=True)
        lstm_handler.fit(loader)
        lstm_handler.save(path_lstm)
        
        # 2. Entrenar GMM (con Anclaje)
        latents = lstm_handler.encode(full_tensor)
        anchor_metric = full_tensor[:, -1, 0].cpu().numpy() # Feature 0 = Volatilidad
        
        logger.info("⚓ Entrenando GMM con ordenamiento semántico...")
        gmm_model.fit(latents, anchor_metric)
        gmm_model.save(path_gmm)
        logger.info("✅ Entrenamiento finalizado.")
        
    else:
        logger.info("🧠 Modelos cargados desde disco.")
        lstm_handler.load(path_lstm)
        gmm_model.load(path_gmm)

    # ---------------------------------------------------------
    # FASE C: INFERENCIA VECTORIZADA (OPTIMIZACIÓN CLAVE)
    # ---------------------------------------------------------
    logger.info("🚀 Ejecutando Inferencia Vectorizada...")
    
    # 1. Inferencia LSTM masiva (Una sola pasada GPU/CPU)
    all_latents = lstm_handler.encode(full_tensor)
    
    # 2. Inferencia GMM masiva
    all_regimes, all_probs = gmm_model.predict(all_latents)
    
    # ---------------------------------------------------------
    # FASE D: RECONSTRUCCIÓN Y MAPEADO
    # ---------------------------------------------------------
    # Ahora debemos "cortar" el array gigante de resultados y dárselo a cada ticker
    results = []
    current_idx = 0
    
    for i, (tk, dates) in enumerate(meta_info):
        n_samples = len(dates)
        
        # Extraemos el slice correspondiente a este ticker
        regime_slice = all_regimes[current_idx : current_idx + n_samples]
        probs_slice = all_probs[current_idx : current_idx + n_samples]
        
        # Avanzamos el puntero
        current_idx += n_samples
        
        # Creamos el DataFrame parcial
        df_res = pl.DataFrame({
            "Date": dates,
            "ticker": tk,
            "market_regime": regime_slice,
            "regime_probability": probs_slice
        })
        
        results.append(df_res)
        
    if not results:
        return pl.DataFrame()

    final_df = pl.concat(results)

    # Filtro final de fecha
    if date:
        target_date = _parse_date(date)
        final_df = final_df.filter(pl.col("Date") == target_date)

    return final_df

def _parse_date(d):
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d")
    return d

if __name__ == "__main__":
    print("--- TEST VECTORIZADO ---")
    try:
        # Test de carga
        df = get_market_regime(tickers=["AAPL", "MSFT", "GOOGL"]) # Varios tickers
        print(df.head())
        print("Regímenes:", df["market_regime"].value_counts())
    except Exception as e:
        print(f"Error: {e}")
