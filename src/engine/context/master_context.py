import sys
import os
import polars as pl
import torch
import numpy as np 
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
    import src.engine.config as config
    from src.engine.src_features import master_features
    from src.engine.context.auto_encoder_lstm import LSTMHandler
    from src.engine.context.gmm_model import RegimeDetector
except ImportError:
    try:
        import config
        from src_features import master_features
        from auto_encoder_lstm import LSTMHandler
        from gmm_model import RegimeDetector
    except ImportError as e:
        print(f"❌ Error CRÍTICO de importación en Context: {e}")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CONTEXT] - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketContext")

def get_market_regime(tickers=None, date=None) -> pl.DataFrame:
    """
    Función maestra optimizada con INFERENCIA VECTORIZADA y PROTECCIÓN ANTI-LOOKAHEAD.
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
    # FASE A: PREPARACIÓN DE DATOS (COMÚN)
    # ---------------------------------------------------------
    unique_tickers = df_input["ticker"].unique().to_list()
    if isinstance(tickers, str): unique_tickers = [tickers]
    elif tickers: unique_tickers = [t for t in unique_tickers if t in tickers]

    tensor_batches = []
    meta_info = [] 
    
    # Lista auxiliar para saber qué índices corresponden a TRAIN (Pasado)
    train_indices_mask = [] 
    
    # FECHA DE CORTE PARA ENTRENAMIENTO (Hard Limit para evitar sesgo)
    # Usamos el 80% del rango de fechas disponible en config o una fecha fija
    # Para simplificar y robustez, usaremos lógica de porcentajes sobre la marcha
    split_ratio = 0.8
    
    logger.info(f"⚙️ Procesando estructuras de tensores para {len(unique_tickers)} tickers...")
    
    current_global_idx = 0
    
    for tk in unique_tickers:
        df_tk = df_input.filter(pl.col("ticker") == tk).sort("Date")
        
        if df_tk.height <= params["LSTM_WINDOW_SIZE"]:
            continue
            
        try:
            t_tk, _ = lstm_handler.prepare_data(df_tk, feat_cols)
            tensor_batches.append(t_tk)
            
            # Metadata
            valid_dates = df_tk["Date"].slice(params["LSTM_WINDOW_SIZE"] - 1, len(df_tk))
            meta_info.append((tk, valid_dates))
            
            # --- MÁSCARA DE ENTRENAMIENTO ---
            # Marcamos como True solo las fechas antiguas (primer 80% de la serie de CADA ticker)
            # Esto asegura que el modelo aprenda de todos los tickers pero solo del pasado.
            n_samples = len(valid_dates)
            n_train = int(n_samples * split_ratio)
            
            # Máscara local: [True, True... (80%), False, False... (20%)]
            mask = torch.zeros(n_samples, dtype=torch.bool)
            mask[:n_train] = True
            train_indices_mask.append(mask)
            
        except ValueError:
            continue

    if not tensor_batches:
        logger.warning("⚠️ Datos insuficientes para generar secuencias válidas.")
        return pl.DataFrame()

    full_tensor = torch.cat(tensor_batches, dim=0)
    full_train_mask = torch.cat(train_indices_mask, dim=0)
    
    logger.info(f"📦 Batch compilado: {full_tensor.shape[0]} ventanas. (Train Set: {full_train_mask.sum()} muestras)")

    # ---------------------------------------------------------
    # FASE B: AUTO-HEALING (ENTRENAMIENTO PROTEGIDO)
    # ---------------------------------------------------------
    if not models_exist:
        logger.warning("⚠️ Modelos no encontrados. Iniciando ENTRENAMIENTO (Split 80/20)...")
        
        # 1. FILTRADO: Solo usamos datos marcados como TRAIN
        train_tensor = full_tensor[full_train_mask]
        
        # 2. Entrenar LSTM (Solo con Train Tensor)
        # Aquí shuffle=True es aceptable porque todo el pool es "Pasado"
        dataset = TensorDataset(train_tensor)
        loader = DataLoader(dataset, batch_size=params["LSTM_BATCH_SIZE"], shuffle=True)
        
        # OJO: Pasamos un loader que SOLO tiene datos de train
        lstm_handler.fit(loader) 
        lstm_handler.save(path_lstm)
        
        # 3. Entrenar GMM (Solo con latentes de Train Tensor)
        # Primero generamos latentes SOLO del set de entrenamiento
        latents_train = lstm_handler.encode(train_tensor)
        anchor_metric_train = train_tensor[:, -1, 0].cpu().numpy() # Volatilidad de Train
        
        logger.info("⚓ Entrenando GMM solo con datos históricos (Train Set)...")
        gmm_model.fit(latents_train, anchor_metric_train)
        gmm_model.save(path_gmm)
        logger.info("✅ Modelos entrenados y guardados sin ver el futuro (Test Set preservado).")
        
    else:
        logger.info("🧠 Modelos cargados desde disco.")
        lstm_handler.load(path_lstm)
        gmm_model.load(path_gmm)

    # ---------------------------------------------------------
    # FASE C: INFERENCIA TOTAL (PRODUCCIÓN)
    # ---------------------------------------------------------
    logger.info("🚀 Ejecutando Inferencia sobre TODO el dataset...")
    
    # Aquí sí procesamos full_tensor (incluyendo el futuro/test) 
    # porque queremos generar señales para todos los días
    all_latents = lstm_handler.encode(full_tensor)
    all_regimes, all_probs = gmm_model.predict(all_latents)
    
    # ---------------------------------------------------------
    # FASE D: RECONSTRUCCIÓN Y MAPEADO
    # ---------------------------------------------------------
    results = []
    current_idx = 0
    
    for i, (tk, dates) in enumerate(meta_info):
        n_samples = len(dates)
        
        regime_slice = all_regimes[current_idx : current_idx + n_samples]
        probs_slice = all_probs[current_idx : current_idx + n_samples]
        
        current_idx += n_samples
        
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
        # Prueba simple
        df = get_market_regime(tickers=["AAPL"])
        print(df.tail()) # Verificamos los últimos datos (que serían Test)
        print("Regímenes:", df["market_regime"].value_counts())
    except Exception as e:
        print(f"Error: {e}")