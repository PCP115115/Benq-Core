# src/engine/config.py


#============================================
#PARÁMETROS GENERALES DEL MOTOR
#============================================
ticker_analizado = "AAPL"
tiempo_analisis = 5
start_date = "2010-01-01"
end_date = "2023-12-31"
layer = "neutral"  # Capas: "all", "robust", "neutral"

#============================================
# --- PARÁMETROS DE INDICADORES (features) ---
#============================================
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
    "YZ_FORECAST_HORIZON": tiempo_analisis,     # Proyección a 5 días (semana bursátil)

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







#==============================================================================
# --- PARÁMETROS DE INTELIGENCIA DEL MÓDULO DE RÉGIMEN DE MERCADO (CONTEXT) ---
#==============================================================================
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
    "FEATURES_OUTPUT": "src/data/processed/features_matrix.parquet",
    
    # Modelos entrenados
    "MODEL_LSTM": "src/data/models/context_lstm.pth",
    "MODEL_GMM": "src/data/models/context_gmm.joblib"
}











#==============================================================================
# --- PARÁMETROS DE MINI-MODELOS (MIXTURE OF EXPERTS) ---
#==============================================================================
MINI_MODEL_PARAMS = {
    "FORECAST_HORIZON": tiempo_analisis,  # Debe coincidir con YZ_FORECAST_HORIZON
    
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

    "MINI_MODEL_TRAIN_PARAMS": {
        "TRAIN_TEST_SPLIT_RATIO": 0.80,  # El porcentaje de corte (0.8 = 80%)
        "TEST_MODE": True,               # True = Hace Split (Backtest), False = Entrena con TODO (Producción)
        "PURGE_OVERLAP": True            # True = Aplica el Purged Gap (recomendado), False = Split simple
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
    ],
    "LAYER" : ["all", "robust", "neutral"]
}

# --- ACTUALIZACIÓN DE RUTAS ---
PATHS.update({
    "MINI_MODELS_DIR": "src/data/models/mini_models/"
})





#============================================
# ---PARÁMETROS META-MODELO---
#============================================
META_MODEL_PARAMS = {

    "start_date": start_date,
    "end_date": end_date,
    "feature_list": ["ker_10", "vol_yz_20d", "amihud_20d", "rsi_14", "macd_line"],
    "normalization_window": 252,  # 1 año bursátil

    "FORECAST_HORIZON": tiempo_analisis,  # Debe coincidir con YZ_FORECAST_HORIZON
    
    "XGB_PARAMS": {
        # --- Configuración del Objetivo ---
        "objective": "binary:logistic", # Salida de probabilidad 0-1 (Se asume un modelo por barrera)
        "eval_metric": "auc",           # Misma métrica que el LightGBM
        "booster": "gbtree",
        "n_jobs": 1,                    # Paralelización externa (como indicaste)
        "random_state": 42,
        
        # --- Control de Sobreajuste (El núcleo de la robustez) ---
        # A diferencia de LightGBM (num_leaves=31), en XGBoost limitamos la profundidad.
        "max_depth": 4,                 
        
        # min_child_weight es CRÍTICO. En LightGBM el default es 20.
        # Aquí forzamos 25: Necesita mucha "evidencia" (muestras) para crear una hoja nueva.
        "min_child_weight": 25,         
        
        # --- Velocidad de Aprendizaje vs Cantidad de Árboles ---
        # Bajamos el learning_rate respecto a tu ejemplo (0.05 -> 0.02) y subimos los estimadores.
        # Esto hace que el modelo aprenda patrones más generales y suaves.
        "learning_rate": 0.02,
        "n_estimators": 500,            # Más árboles para compensar el learning rate bajo (usar early stopping).
        
        # --- Gestión de los Expertos (MoE) y Regímenes ---
        # colsample_bytree=0.6 obliga al árbol a mirar solo el 60% de los expertos cada vez.
        # Esto evita que el modelo dependa siempre del mismo "Super Experto" que podría fallar en el futuro.
        "colsample_bytree": 0.6,
        
        # subsample=0.7 entrena cada árbol con el 70% de los datos (bagging).
        "subsample": 0.7,
        
        # --- Regularización (Castigo a la complejidad) ---
        # LightGBM usa l1/l2 por defecto en 0. Aquí los forzamos.
        # reg_alpha (L1): Pone a CERO el peso de expertos inútiles (selección de features).
        "reg_alpha": 1.5,
        # reg_lambda (L2): Suaviza los pesos para evitar que una predicción se dispare a 1.0 o 0.0 fácilmente.
        "reg_lambda": 5.0,
        
        # --- Parámetro de "Poda" ---
        # gamma: Mínima reducción de pérdida para hacer una división. 
        # Un valor > 0 hace al algoritmo conservador.
        "gamma": 0.2
    },

    "META_MODEL_TRAIN_PARAMS": {
        "TRAIN_TEST_SPLIT_RATIO": 0.80,
        "TEST_MODE": True,
        "PURGE_OVERLAP": True 
    }
}