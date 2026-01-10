# src/engine/config.py

# --- PARÁMETROS DE INDICADORES ---
FEATURES_PARAMS = {
    # Ventanas Temporales Básicas
    "RSI_PERIOD": 14,             # Estándar de Wilder
    "VOLATILITY_WINDOW": 20,      # 1 Mes bursátil (std dev de retornos)
    "SKEW_WINDOW": 60,            # ~3 Meses (Detectar riesgo de cola trimestral)
    "CORR_WINDOW": 20,            # Correlación Precio-Volumen mensual
    "KER_WINDOW": 10,             # Eficiencia de Kaufman (2 semanas)
    
    # Volatilidad Avanzada (Range-Based)
    "PARKINSON_WINDOW": 20,       # Aumentado a 20 para alinear con el mes bursátil
    "GARMAN_KLASS_WINDOW": 20,    # [NUEVO] Para la nueva función GK. 20 es estándar industrial.
    "YANG_ZHANG_WINDOW": 20,      # [NUEVO] Estimador Yang-Zhang (Drift independent + Gaps)
    
    # [NUEVO] Conos de Volatilidad (Forecast Bounds)
    "YZ_Z_SCORE": 1.96,           # Intervalo de confianza 95%
    "YZ_FORECAST_HORIZON": 5,     # Proyección a 5 días (semana bursátil)

    # Liquidez
    "AMIHUD_WINDOW": 20,          # Ventana de media móvil para iliquidez
    "AMIHUD_SCALING": 1e6,        # [NUEVO] Sacado de la función. Permite ajustar según el activo.
    
    # Medias Móviles para Tendencia Relativa
    "SMA_FAST": 15,
    "SMA_MEDIUM": 30,
    "SMA_SLOW": 50,
    
    # MACD (Estándar)
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9
}

# --- PARÁMETROS DE NORMALIZACIÓN ---
NORMALIZATION_PARAMS = {
    # Ventana para el Robust Scaler Temporal (Rolling)
    # 252 días = 1 Año bursátil. 
    "ROLLING_WINDOW": 252, 
    
    # Mínimo de datos para z-score (evita ruido al inicio de la serie)
    "MIN_PERIODS": 120,
    
    # Neutralización Sectorial
    "MIN_ASSETS_PER_SECTOR": 5
}

# --- RUTAS DEL PROYECTO (NUEVO) ---
# Define dónde se guardarán los outputs. 
# La ruta es relativa al root del proyecto.
PATHS = {
    "FEATURES_OUTPUT": "data/processed/features_matrix.parquet"
}



# --- PARÁMETROS DE INTELIGENCIA DEL MÓDULO DE RÉGIMEN DE MERCADO (CONTEXT) ---
CONTEXT_PARAMS = {
    # Features de entrada para el Autoencoder (Deben existir en indicators.py)
    # Seleccionamos: Volatilidad, Eficiencia, Liquidez y Correlación
    "INPUT_FEATURES": [
        "vol_yz_20d", 
        "ker_10", 
        "amihud_20d", 
        "corr_price_vol_20d"
    ],
    
    "LSTM_WINDOW_SIZE": 20,       # Lookback window (T) para la red neuronal
    "NORMALIZATION_WINDOW": 63,   # Ventana trimestral para el Robust Scaler dinámico
    
    # Configuración del Modelo
    "LSTM_HIDDEN_DIM": 32,        # Dimensión de las neuronas
    "LSTM_LATENT_DIM": 10,        # Dimensión comprimida (Features latentes)
    "LSTM_LAYERS": 1,
    "LSTM_EPOCHS": 50,
    "LSTM_BATCH_SIZE": 64,
    "LSTM_LR": 1e-3,
    
    "GMM_N_COMPONENTS": 5,        # Número de Regímenes de Mercado
    "GMM_COVARIANCE_TYPE": "full"
}

# --- RUTAS ACTUALIZADAS ---
PATHS = {
    "FEATURES_OUTPUT": "data/processed/features_matrix.parquet",
    
    # Modelos entrenados
    "MODEL_LSTM": "data/models/context_lstm.pth",
    "MODEL_GMM": "data/models/context_gmm.joblib"
}

# --- PARÁMETROS DE MINI-MODELOS (MIXTURE OF EXPERTS) ---
MINI_MODEL_PARAMS = {
    "FORECAST_HORIZON": 5,  # Debe coincidir con YZ_FORECAST_HORIZON
    
    # Hiperparámetros base para LightGBM
    "LGBM_PARAMS": {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": 1  # 1 hilo por modelo, ya que paralelizamos a nivel de proceso
    },
    
    # Definición de Features por Experto (Nombres base, sin sufijo _rob)
    "FEATURES_TREND": [
        "adx_14", 
        "macd_line", 
        "macd_hist", 
        "rel_sma_15", 
        "rel_sma_50", 
        "ker_10", 
        "corr_price_vol_20d"
    ],
    
    "FEATURES_VOLATILITY": [
        "vol_yz_20d", 
        "vol_gk_20d", 
        "vol_parkinson_20d", 
        "vol_std_20d", 
        "amihud_20d"
    ],
    
    "FEATURES_REVERSION": [
        "rsi_14", 
        "skew_60d", 
        "rel_sma_15", 
        "macd_hist"
    ]
}

# --- ACTUALIZACIÓN DE RUTAS ---
PATHS.update({
    "MINI_MODELS_DIR": "data/models/mini_models/"
})