# src/strategy/config_strategy.py

"""
Configuración centralizada para el Bloque de Estrategia.
"""

# --- LISTA DE ACTIVOS ---
TICKERS_ESTRATEGIA = [
    "AAPL", "TSLA", "BAC", "JPM", "AMZN", "X", "CAT", "SAN"
]

# --- CONFIGURACIÓN DEL META-MODELO ---
META_MODEL_CONFIG = {
    "FORCE_RETRAIN": False,
    "FORECAST_HORIZON": 5,        # Horizonte del modelo (días)
    "YZ_Z_SCORE": 1.96,
    "VOL_WINDOW": 20,
    "MIN_PROB_THRESHOLD": 0.51
}

# --- CONFIGURACIÓN DE BLACK-LITTERMAN ---
BLACK_LITTERMAN_CONFIG = {
    # Horizonte de Optimización (Días): Escala las predicciones a este periodo
    "OPTIMIZATION_HORIZON": 5, 
    "RISK_AVERSION": 3.0,
    "TAU": 0.05
}

# --- CONFIGURACIÓN DE OPTIMIZACIÓN DE CARTERA (NUEVO) ---
PORTFOLIO_CONFIG = {
    # Objetivo de la Optimización:
    # "MAX_SHARPE"      -> Maximizar rentabilidad ajustada al riesgo (Recomendado).
    # "MAX_RETURN"      -> Maximizar rentabilidad bruta (Mayor riesgo).
    # "MIN_VOLATILITY"  -> Minimizar riesgo absoluto (Cartera defensiva).
    "OBJECTIVE": "MAX_SHARPE",

    # Tasa Libre de Riesgo Anual (Risk Free Rate)
    # Se usa para el cálculo del Sharpe Ratio (ej. 0.04 = 4% anual).
    # El script la escalará automáticamente al horizonte de días configurado.
    "RISK_FREE_RATE_ANNUAL": 0.04,

    # --- Restricciones (Constraints) ---
    # Permitir posiciones cortas (Venta en corto)
    # True: Permite pesos negativos. False: Solo compras (Long-only).
    "ALLOW_SHORTS": True,

    # Peso Máximo por Activo (0.0 a 1.0)
    # Ej: 0.40 significa que ningún activo puede superar el 40% de la cartera.
    "MAX_WEIGHT_PER_ASSET": 0.40,

    # Peso Mínimo por Activo
    # Si ALLOW_SHORTS es False, esto suele ser 0.0.
    # Si ALLOW_SHORTS es True, esto puede ser negativo (ej. -0.20 para max corto del 20%).
    "MIN_WEIGHT_PER_ASSET": -0.20
}

# --- OUTPUT ---
OUTPUT_CONFIG = {
    "EXPORT_TO_CSV": False,
    "OUTPUT_FILENAME": "strategy_signals.csv"
}


SIZING_CONFIG = {
    # Capital Total de la Cuenta (en la divisa base, ej. USD/EUR)
    # El script calculará cantidades exactas basadas en este número.
    "TOTAL_CAPITAL": 100_000.0,

    # Volatilidad Anual Objetivo (Volatility Target)
    # 0.12 = 12% anual (Tu elección: Moderado-Agresivo).
    # Si la cartera es más tranquila que esto, nos apalancamos.
    # Si es más nerviosa, reducimos exposición.
    "TARGET_VOLATILITY_ANNUAL": 0.20,

    # Apalancamiento Máximo (Leverage Cap)
    # 1.5 = Permitir hasta 150% de exposición (1.5x) si la volatilidad es baja.
    "MAX_LEVERAGE": 1.5,
    
    # Colchón de liquidez mínimo (Cash Buffer)
    # 0.02 = Dejar siempre al menos un 2% en efectivo para comisiones/deslizamientos.
    "MIN_CASH_BUFFER": 0.02
}



EXECUTION_CONFIG = {
    # Valor mínimo de una orden en divisa base para ser enviada
    # Evita órdenes de $10 que solo generan comisiones.
    "MIN_ORDER_VALUE": 100.0, 

    # Si es True, el script se detiene si detecta que es Fin de Semana.
    "CHECK_MARKET_OPEN": False,
    
    # Carpeta donde se guardarán los CSVs de órdenes
    "ORDERS_DIR": "orders"
}

# --- OUTPUT ---
OUTPUT_CONFIG = {
    "EXPORT_TO_CSV": True, # Forzamos True para el Master
    "OUTPUT_FILENAME": "strategy_signals.csv" # Nombre base (se añadirá fecha)
}




# --- CONFIGURACIÓN DE BACKTESTING ---
BACKTEST_CONFIG = {
    # Periodo: 2023-2025 (2 años sólidos es suficiente para validar)
    "START_DATE": "2020-01-01",
    "END_DATE": "2025-12-31",
    "INITIAL_CAPITAL": 100_000.0,
    
    # Frecuencia de Rebalanceo OPTIMIZADA:
    # "MS": Month Start (Primer día del mes). 
    # Esto es mucho más rápido que Semanal ("W-FRI").
    "REBALANCE_FREQ": "MS", 
    
    "COMMISSION_PCT": 0.0010,
    "SLIPPAGE_PCT": 0.0005,
    "BENCHMARK_TICKER": "SPY" 
}