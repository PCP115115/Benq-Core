import polars as pl
import numpy as np

# Constantes matemáticas
LOG_2 = np.log(2)
PARKINSON_CONST = 1.0 / (4.0 * LOG_2)
# Constante para Garman-Klass: (2 * ln(2) - 1)
GK_CONST = (2.0 * LOG_2) - 1.0

def get_log_returns(col_name: str = "Close") -> pl.Expr:
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
    Desviación típica rodante del volumen NORMALIZADA (Coeficiente de Variación).
    Formula: Rolling_Std / Rolling_Mean
    Mantiene el mismo alias para compatibilidad.
    """
    roll_std = pl.col(col_vol).rolling_std(window_size=window)
    roll_mean = pl.col(col_vol).rolling_mean(window_size=window)
    
    return (
        (roll_std / roll_mean)
        .fill_nan(0.0)
        .alias(f"vol_std_{window}d")
    )

def get_rolling_skewness(col_returns: str, window: int) -> pl.Expr:
    """
    Asimetría rodante (Rolling Skewness).
    """
    return (
        pl.col(col_returns)
        .rolling_skew(window_size=window)
        .alias(f"skew_{window}d")
    )

def get_volume_return_correlation(col_returns: str, col_vol: str, window: int) -> pl.Expr:
    """
    Correlación rodante entre precio (retornos) y volumen.
    """
    return (
        pl.rolling_corr(pl.col(col_returns), pl.col(col_vol), window_size=window)
        .alias(f"corr_price_vol_{window}d")
    )

def get_relative_sma(col_name: str, window: int) -> pl.Expr:
    """
    Distancia relativa a la media móvil: (Precio - SMA) / SMA.
    """
    sma = pl.col(col_name).rolling_mean(window_size=window)
    return (
        ((pl.col(col_name) - sma) / sma)
        .alias(f"rel_sma_{window}")
    )

def get_efficiency_ratio_ker(col_name: str, window: int) -> pl.Expr:
    """
    Kaufman Efficiency Ratio (KER).
    """
    change_net = (pl.col(col_name) - pl.col(col_name).shift(window)).abs()
    change_sum = (pl.col(col_name).diff().abs().rolling_sum(window_size=window))
    
    return (
        (change_net / change_sum)
        .fill_nan(0.0)
        .alias(f"ker_{window}")
    )

def get_parkinson_volatility(col_high: str, col_low: str, window: int) -> pl.Expr:
    """
    Volatilidad de Parkinson (High-Low).
    Mantenida por compatibilidad. 
    Formula: sqrt( (1 / 4*ln(2)) * rolling_mean( ln(H/L)^2 ) )
    """
    log_hl_sq = (pl.col(col_high) / pl.col(col_low)).log().pow(2)
    
    return (
        (log_hl_sq.rolling_mean(window_size=window) * PARKINSON_CONST)
        .sqrt()
        .alias(f"vol_parkinson_{window}d")
    )

def get_garman_klass_volatility(col_high: str, col_low: str, col_close: str, col_open: str, window: int) -> pl.Expr:
    """
    Volatilidad de Garman-Klass.
    Extensión de Parkinson que incluye información de apertura y cierre.
    Más eficiente (menor varianza en la estimación) que Parkinson y Close-to-Close.
    Formula: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    """
    # Usamos log logs pre-calculados para eficiencia vectorizada
    log_hl_sq = (pl.col(col_high) / pl.col(col_low)).log().pow(2)
    log_co_sq = (pl.col(col_close) / pl.col(col_open)).log().pow(2)
    
    # Expresión interna de GK
    gk_estimator = (0.5 * log_hl_sq) - (GK_CONST * log_co_sq)
    
    return (
        gk_estimator.rolling_mean(window_size=window)
        .sqrt()
        .alias(f"vol_gk_{window}d")
    )

def get_yang_zhang_volatility(col_open: str, col_high: str, col_low: str, col_close: str, window: int) -> pl.Expr:
    """
    Volatilidad de Yang-Zhang (Drift Independent + Gaps).
    Combina volatilidad Overnight, Open-Close y Rogers-Satchell.
    Formula: sqrt( Var_Overnight + k * Var_OpenClose + (1-k) * Var_RS )
    """
    # 1. Cálculo de k dinámico
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    
    # 2. Logaritmos base
    log_o = pl.col(col_open).log()
    log_h = pl.col(col_high).log()
    log_l = pl.col(col_low).log()
    log_c = pl.col(col_close).log()
    log_c_prev = log_c.shift(1)
    
    # 3. Componente Overnight: Varianza de ln(Open_t / Close_t-1)
    # drift-independent implica usar varianza muestral
    term_overnight_var = (log_o - log_c_prev).rolling_var(window_size=window)
    
    # 4. Componente Open-Close: Varianza de ln(Close_t / Open_t)
    term_open_close_var = (log_c - log_o).rolling_var(window_size=window)
    
    # 5. Componente Rogers-Satchell (RS): Media rodante
    # RS = ln(H/C)ln(H/O) + ln(L/C)ln(L/O)
    rs_raw = (
        ((log_h - log_c) * (log_h - log_o)) + 
        ((log_l - log_c) * (log_l - log_o))
    )
    term_rs_var = rs_raw.rolling_mean(window_size=window)
    
    # 6. Combinación Ponderada
    yz_variance = (
        term_overnight_var + 
        (k * term_open_close_var) + 
        ((1.0 - k) * term_rs_var)
    )
    
    return (
        yz_variance.sqrt()
        .alias(f"vol_yz_{window}d")
    )

def get_volatility_bounds(col_close: str, col_vol_yz: str, z_score: float, horizon: int) -> list[pl.Expr]:
    """
    Genera Conos de Volatilidad (Techo y Suelo) basados en la proyección de Yang-Zhang.
    Formula: Price * (1 +/- Z * Vol * sqrt(T))
    """
    # Factor de proyección: Volatilidad * Z * sqrt(Tiempo)
    # Importante: np.sqrt es escalar aquí, lo cual es eficiente.
    projection_factor = pl.col(col_vol_yz) * z_score * np.sqrt(horizon)
    
    ceil = (pl.col(col_close) * (1 + projection_factor)).alias(f"fprice_ceil_yz_{horizon}d")
    floor = (pl.col(col_close) * (1 - projection_factor)).alias(f"fprice_floor_yz_{horizon}d")
    
    return [ceil, floor]

def get_amihud_liquidity(col_abs_ret: str, col_price: str, col_vol: str, window: int, scaling_factor: float = 1e6) -> pl.Expr:
    """
    Iliquidez de Amihud con transformación LOGARÍTMICA.
    Ayuda a comprimir los outliers extremos y mejora la estacionariedad.
    """
    # Cálculo original
    daily_illiquidity = (
        pl.col(col_abs_ret) / (pl.col(col_price) * pl.col(col_vol))
    )
    
    # Promedio rodante
    rolling_amihud = daily_illiquidity.rolling_mean(window_size=window) * scaling_factor
    
    return (
        rolling_amihud
        .log1p() 
        .alias(f"amihud_{window}d")
    )

def get_rsi(col_name: str, period: int) -> pl.Expr:
    """
    Relative Strength Index (RSI).
    ACTUALIZADO: Usa 'com' en lugar de 'span' para replicar Wilder's Smoothing.
    CORREGIDO: Usa 'min_samples' en lugar de 'min_periods' (Deprecation Fix).
    alpha = 1 / period  <=> com = period - 1
    """
    delta = pl.col(col_name).diff()
    up = delta.clip(lower_bound=0)
    down = delta.clip(upper_bound=0).abs()
    

    wilder_com = period - 1
    
    # Fix: min_periods -> min_samples
    roll_up = up.ewm_mean(min_samples=period, adjust=False, com=wilder_com)
    roll_down = down.ewm_mean(min_samples=period, adjust=False, com=wilder_com)
    
    rs = roll_up / roll_down
    return (
        (100.0 - (100.0 / (1.0 + rs)))
        .alias(f"rsi_{period}")
    )

def get_macd_expressions(col_name: str, fast: int, slow: int, signal: int) -> list[pl.Expr]:
    """
    Versión NORMALIZADA (PPO - Percentage Price Oscillator).
    Calcula la diferencia como porcentaje de la EMA lenta.
    """
    ema_fast = pl.col(col_name).ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col(col_name).ewm_mean(span=slow, adjust=False)
    
    # CAMBIO CLAVE: ((Fast - Slow) / Slow) * 100
    macd_line = (((ema_fast - ema_slow) / ema_slow) * 100).alias("macd_line")
    
    # La señal se calcula sobre la línea ya normalizada
    macd_signal = macd_line.ewm_mean(span=signal, adjust=False).alias("macd_signal")
    
    # El histograma mantiene la escala porcentual
    macd_hist = (macd_line - macd_signal).alias("macd_hist")
    
    return [macd_line, macd_signal, macd_hist]

def get_adx(col_high: str, col_low: str, col_close: str, period: int = 14) -> list[pl.Expr]:
    """
    Average Directional Index (ADX) con suavizado de Wilder.
    Devuelve [adx, plus_di, minus_di].
    """
    # 1. True Range (TR)
    # TR = Max(H-L, Abs(H-Cp), Abs(L-Cp))
    tr1 = pl.col(col_high) - pl.col(col_low)
    tr2 = (pl.col(col_high) - pl.col(col_close).shift(1)).abs()
    tr3 = (pl.col(col_low) - pl.col(col_close).shift(1)).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)

    # 2. Directional Movement (DM)
    up_move = pl.col(col_high) - pl.col(col_high).shift(1)
    down_move = pl.col(col_low).shift(1) - pl.col(col_low)

    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)

    # 3. Suavizado de Wilder (alpha = 1/n  => com = n - 1)
    wilder_com = period - 1
    
    tr_smooth = tr.ewm_mean(com=wilder_com, adjust=False, min_samples=period)
    plus_dm_smooth = plus_dm.ewm_mean(com=wilder_com, adjust=False, min_samples=period)
    minus_dm_smooth = minus_dm.ewm_mean(com=wilder_com, adjust=False, min_samples=period)

    # 4. Directional Indicators (DI)
    plus_di = (100 * plus_dm_smooth / tr_smooth).alias(f"plus_di_{period}")
    minus_di = (100 * minus_dm_smooth / tr_smooth).alias(f"minus_di_{period}")

    # 5. DX y ADX
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs())
    adx = dx.ewm_mean(com=wilder_com, adjust=False, min_samples=period).alias(f"adx_{period}")

    return [adx, plus_di, minus_di]