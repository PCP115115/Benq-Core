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
