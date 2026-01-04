import polars as pl
import logging
import sys
import os

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [FEATURES] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureEngine")

# --- GESTIÓN DE IMPORTACIONES ---
try:
    from tickers import FEATURES_PARAMS
    from loader import MarketLoader
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tickers import FEATURES_PARAMS
    from loader import MarketLoader

class FeatureEngine:
    def __init__(self):
        self.params = FEATURES_PARAMS
        # Inicializamos el Loader con DuckDB
        self.loader = MarketLoader(actualizar_datos=False)
        
    def _get_indicators_expr(self) -> list[pl.Expr]:
        """
        Define el grafo de computación para los indicadores técnicos.
        """
        col_close = pl.col("Close")
        col_vol = pl.col("Volume")

        # --- 1. Retornos y Volatilidad ---
        log_ret = (col_close / col_close.shift(1)).log().alias("Log_Returns")
        volatility = log_ret.rolling_std(window_size=self.params["VOLATILITY_WINDOW"]).alias("Volatility")

        # --- 2. Tendencias Relativas ---
        sma_trend = col_close.rolling_mean(window_size=self.params["TREND_WINDOW"])
        rel_trend = ((col_close - sma_trend) / sma_trend).alias("Relative_Trend")

        sma_vol = col_vol.rolling_mean(window_size=self.params["REL_VOL_WINDOW"])
        rel_vol = (col_vol / sma_vol).alias("Relative_Volume")

        # --- 3. RSI ---
        delta = col_close.diff()
        up = delta.clip(lower_bound=0)
        down = delta.clip(upper_bound=0).abs()
        alpha = 1 / self.params["RSI_PERIOD"]
        
        roll_up = up.ewm_mean(alpha=alpha, adjust=False, min_periods=self.params["RSI_PERIOD"])
        roll_down = down.ewm_mean(alpha=alpha, adjust=False, min_periods=self.params["RSI_PERIOD"])
        rs_ratio = roll_up / roll_down
        rsi = (100 - (100 / (1 + rs_ratio))).alias("RSI")

        # --- 4. MACD ---
        ema_fast = col_close.ewm_mean(span=self.params["MACD_FAST"], adjust=False)
        ema_slow = col_close.ewm_mean(span=self.params["MACD_SLOW"], adjust=False)
        macd_line = (ema_fast - ema_slow).alias("MACD_Line")
        macd_signal = macd_line.ewm_mean(span=self.params["MACD_SIGNAL"], adjust=False).alias("MACD_Signal")
        macd_hist = (macd_line - macd_signal).alias("MACD_Hist")

        # --- 5. NUEVAS VARIABLES ---
        
        # A) EWNA
        ewna = col_close.ewm_mean(span=self.params["TREND_WINDOW"], adjust=False).alias("EWNA")

        # B) Rolling Z-Score
        roll_mean_z = col_close.rolling_mean(window_size=self.params["VOLATILITY_WINDOW"])
        roll_std_z = col_close.rolling_std(window_size=self.params["VOLATILITY_WINDOW"])
        z_score = ((col_close - roll_mean_z) / roll_std_z).alias("Rolling_Z_Score")

        # C) Efficiency Ratio KER
        ker_window = 10 
        change_net = (col_close - col_close.shift(ker_window)).abs()
        change_sum = (col_close - col_close.shift(1)).abs().rolling_sum(window_size=ker_window)
        efficiency_ratio = (change_net / change_sum).fill_nan(0).alias("Efficiency_Ratio_KER")

        return [
            log_ret, volatility, rel_trend, rel_vol, 
            rsi, macd_line, macd_signal, macd_hist,
            ewna, z_score, efficiency_ratio
        ]

    def get_market_dataset(self, sector: str = None) -> pl.DataFrame:
        """
        Orquesta la carga y cálculo usando DuckDB -> Polars.
        """
        logger.info("🚀 Iniciando Feature Pipeline (via DuckDB)...")

        # 1. Extracción de Datos (SQL Pushdown) [Diagram of Data Flow: Parquet -> DuckDB -> Arrow -> Polars]
        # En lugar de leer archivos, pedimos el dataset al Loader
        try:
            if sector:
                logger.info(f"Filtrando por sector: {sector}")
                # El loader ya devuelve un Polars DataFrame
                df_base = self.loader.query(f"SELECT * FROM market WHERE sector = '{sector}'")
            else:
                # Todo el mercado
                df_base = self.loader.get_all_data()

            if df_base.is_empty():
                logger.critical("El dataset base está vacío.")
                return None

        except Exception as e:
            logger.critical(f"Error extrayendo datos del loader: {e}")
            return None

        # 2. Preprocesamiento & Casting
        # DuckDB nos da los tipos inferidos, pero aseguramos float64 y manejamos data_quality
        try:
            # Si data_quality no existe o tiene nulos (por union_by_name), rellenamos con 1
            quality_expr = pl.col("data_quality").fill_null(1).cast(pl.Int8) if "data_quality" in df_base.columns else pl.lit(1, dtype=pl.Int8).alias("data_quality")

            lf_base = (
                df_base.lazy()
                .with_columns([
                    pl.col("Open").cast(pl.Float64),
                    pl.col("High").cast(pl.Float64),
                    pl.col("Low").cast(pl.Float64),
                    pl.col("Close").cast(pl.Float64),
                    pl.col("Volume").cast(pl.Int64),
                    quality_expr
                ])
            )
        except Exception as e:
            logger.critical(f"Error en casting de columnas: {e}")
            return None

        # 3. Cálculo de Indicadores (Igual que antes, pero sobre el LazyFrame unificado)
        logger.info("Calculando indicadores técnicos (Engine: Polars Rust)...")
        
        lf_enriched = (
            lf_base
            .sort(["ticker", "Date"])
            .with_columns([
                expr.over("ticker") for expr in self._get_indicators_expr()
            ])
            .drop_nulls()
        )

        # 4. Materialización
        try:
            df_final = lf_enriched.collect()
            logger.info(f"✅ Pipeline finalizado. Shape: {df_final.shape}")
            return df_final
            
        except Exception as e:
            logger.error(f"Error durante la materialización: {e}")
            return None

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    engine = FeatureEngine()
    
    # Ejemplo: Podemos pedir solo un sector para probar
    df = engine.get_market_dataset(sector=None) # None = Todos
    
    if df is not None:
        print(f"⏱️ Tiempo total Pipeline: {time.perf_counter() - start:.4f}s")
        if "Efficiency_Ratio_KER" in df.columns:
             print("Test nuevas variables: OK")
             print(df.head(5))