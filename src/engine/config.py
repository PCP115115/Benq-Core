# src/engine/config.py

# --- PARÁMETROS DE INDICADORES ---
FEATURES_PARAMS = {
    # Ventanas Temporales
    "RSI_PERIOD": 14,
    "VOLATILITY_WINDOW": 20,      # 1 Mes bursátil
    "SKEW_WINDOW": 60,            # ~3 Meses (Trimestral)
    "CORR_WINDOW": 20,
    "KER_WINDOW": 10,             # 2 Semanas
    "PARKINSON_WINDOW": 14,
    "AMIHUD_WINDOW": 20,
    
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
    # 252 días = 1 Año bursátil. Comparamos el dato de hoy con el último año del activo.
    "ROLLING_WINDOW": 252, 
    
    # Mínimo de datos necesarios para calcular z-score (evita ruido al inicio)
    "MIN_PERIODS": 120,
    
    # Mínimo de acciones en un sector para neutralizar (evita grupos vacíos)
    "MIN_ASSETS_PER_SECTOR": 5
}

# Rutas de Salida
PATHS = {
    "FEATURES_OUTPUT": "data/features/features_matrix.parquet"
}