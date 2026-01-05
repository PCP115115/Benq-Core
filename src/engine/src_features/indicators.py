import polars as pl
import numpy as np

# Constantes matemáticas
LOG_2 = np.log(2)
PARKINSON_CONST = 1.0 / (4.0 * LOG_2)

def get_log_returns(col_name: str = "Close") -> pl.Expr:
    """
    Calcula los retornos logarítmicos: ln(Pt / Pt-1).
    Es preferible a los retornos simples por su propiedad de aditividad en el tiempo.
    """
    return (
        (pl.col(col_name) / pl.col(col_name).shift(1))
        .log()
        .alias("log_returns")
    )

def get_rolling_volatility(col_returns: str, window: int) -> pl.Expr:
    """
    Desviación típica rodante de los retornos (Volatilidad histórica).
    """
    return (
        pl.col(col_returns)
        .rolling_std(window_size=window)
        .alias(f"volatility_{window}d")
    )

def get_volume_std(col_vol: str, window: int) -> pl.Expr:
    """
    Desviación típica rodante del volumen.
    Útil para detectar regímenes de actividad anómala.
    """
    return (
        pl.col(col_vol)
        .rolling_std(window_size=window)
        .alias(f"vol_std_{window}d")
    )

def get_rolling_skewness(col_returns: str, window: int) -> pl.Expr:
    """
    Asimetría rodante (Rolling Skewness).
    Detecta riesgo de cola (Crash Risk). Skew negativo alto = peligro.
    """
    return (
        pl.col(col_returns)
        .rolling_skew(window_size=window)
        .alias(f"skew_{window}d")
    )

def get_volume_return_correlation(col_returns: str, col_vol: str, window: int) -> pl.Expr:
    """
    Correlación rodante entre precio (retornos) y volumen.
    Teoría: Las subidas con volumen confirman tendencia.
    """
    return (
        pl.rolling_corr(pl.col(col_returns), pl.col(col_vol), window_size=window)
        .alias(f"corr_price_vol_{window}d")
    )

def get_relative_sma(col_name: str, window: int) -> pl.Expr:
    """
    Distancia relativa a la media móvil: (Precio - SMA) / SMA.
    Normaliza la posición del precio respecto a su tendencia, haciéndola comparable entre activos.
    """
    sma = pl.col(col_name).rolling_mean(window_size=window)
    return (
        ((pl.col(col_name) - sma) / sma)
        .alias(f"rel_sma_{window}")
    )

def get_efficiency_ratio_ker(col_name: str, window: int) -> pl.Expr:
    """
    Kaufman Efficiency Ratio (KER).
    Mide la "limpieza" de la tendencia.
    KER = |Cambio Neto| / Suma(|Cambios Individuales|)
    Rango: 0 (Ruido total) a 1 (Tendencia perfecta).
    """
    change_net = (pl.col(col_name) - pl.col(col_name).shift(window)).abs()
    change_sum = (pl.col(col_name).diff().abs().rolling_sum(window_size=window))
    
    # Evitar división por cero rellenando con muy pequeño o 1 si es 0
    return (
        (change_net / change_sum)
        .fill_nan(0.0)
        .alias(f"ker_{window}")
    )

def get_parkinson_volatility(col_high: str, col_low: str, window: int) -> pl.Expr:
    """
    Volatilidad de Parkinson (High-Low).
    Más eficiente que la desviación típica de cierre porque usa el rango diario.
    Formula: sqrt( (1 / 4*ln(2)) * rolling_mean( ln(H/L)^2 ) )
    """
    log_hl_sq = (pl.col(col_high) / pl.col(col_low)).log().pow(2)
    
    return (
        (log_hl_sq.rolling_mean(window_size=window) * PARKINSON_CONST)
        .sqrt()
        .alias(f"vol_parkinson_{window}d")
    )

def get_amihud_liquidity(col_abs_ret: str, col_price: str, col_vol: str, window: int, scaling_factor: float = 1e6) -> pl.Expr:
    """
    Iliquidez de Amihud (Proxy de Impacto en Mercado).
    Formula: |Retorno| / (Precio * Volumen)
    Interpretación: Valores ALTOS = Iliquidez (el precio se mueve mucho con poco volumen).
    Factor de escala: Para evitar números infinitesimales (e.g. 1e-9).
    """
    daily_illiquidity = (
        pl.col(col_abs_ret) / (pl.col(col_price) * pl.col(col_vol))
    )
    
    return (
        (daily_illiquidity.rolling_mean(window_size=window) * scaling_factor)
        .alias(f"amihud_{window}d")
    )

def get_rsi(col_name: str, period: int) -> pl.Expr:
    """
    Relative Strength Index (RSI).
    Implementación vectorizada manual para Polars.
    """
    delta = pl.col(col_name).diff()
    up = delta.clip(lower_bound=0)
    down = delta.clip(upper_bound=0).abs()
    
    # Media Móvil Exponencial (Wilder's Smoothing es similar a una EWM con alpha=1/period)
    # Nota: Usamos ewm_mean de Polars con adjust=False para replicar mejor el comportamiento clásico
    roll_up = up.ewm_mean(min_periods=period, adjust=False, span=period)
    roll_down = down.ewm_mean(min_periods=period, adjust=False, span=period)
    
    rs = roll_up / roll_down
    return (
        (100.0 - (100.0 / (1.0 + rs)))
        .alias(f"rsi_{period}")
    )

def get_macd_expressions(col_name: str, fast: int, slow: int, signal: int) -> list[pl.Expr]:
    """
    Devuelve TRES expresiones para generar las columnas del MACD:
    1. MACD Line
    2. MACD Signal
    3. MACD Histogram
    """
    # Cálculos intermedios (Lazy, Polars optimizará si se repiten)
    ema_fast = pl.col(col_name).ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col(col_name).ewm_mean(span=slow, adjust=False)
    
    macd_line = (ema_fast - ema_slow).alias("macd_line")
    macd_signal = macd_line.ewm_mean(span=signal, adjust=False).alias("macd_signal")
    macd_hist = (macd_line - macd_signal).alias("macd_hist")
    
    return [macd_line, macd_signal, macd_hist]
