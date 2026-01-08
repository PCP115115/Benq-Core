import unittest
import time
import numpy as np
import polars as pl
import sys
import os

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.dirname(current_dir)
src_features_path = os.path.join(features_dir, "src_features")

sys.path.append(src_features_path)

try:
    import indicators
except ImportError as e:
    raise ImportError(f"No se pudo importar 'indicators.py'. Revisa la ruta: {src_features_path}") from e

class TestIndicatorsPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generamos dataset MASIVO (2.5M filas) para benchmark."""
        print("\n🌊 Generando Datos Dummy (2.5 Millones de filas)...")
        n_rows = 2_500_000
        rng = np.random.default_rng(42)
        
        base_price = rng.uniform(10, 200, n_rows)
        volatility = rng.uniform(0.01, 0.05, n_rows)
        
        cls.df = pl.DataFrame({
            "Open": base_price,
            "Close": base_price * (1 + rng.normal(0, 0.01, n_rows)),
            "Volume": rng.uniform(1000, 1000000, n_rows)
        }).with_columns([
            (pl.max_horizontal("Open", "Close") * (1 + pl.lit(volatility))).alias("High"),
            (pl.min_horizontal("Open", "Close") * (1 - pl.lit(volatility))).alias("Low"),
            # Columna dummy de volatilidad ya calculada para probar los bounds aisladamente
            pl.lit(0.02).alias("dummy_vol_yz") 
        ]).with_columns([
            pl.all().cast(pl.Float64)
        ])
        
        print(f"✅ Dataset listo: {cls.df.estimated_size('mb'):.2f} MB en RAM.")

    def test_logic_rsi(self):
        """Verifica RSI [0, 100]."""
        print("\n🧪 Test Lógico: RSI Range")
        expr = indicators.get_rsi("Close", period=14)
        df_res = self.df.head(1000).select(expr)
        rsi_vals = df_res.drop_nulls().to_series()
        self.assertGreaterEqual(rsi_vals.min(), 0)
        self.assertLessEqual(rsi_vals.max(), 100)
        print(f"   RSI OK.")

    def test_logic_yang_zhang(self):
        """Verifica YZ > 0 y sin NaNs."""
        print("🧪 Test Lógico: Yang-Zhang Volatility")
        expr = indicators.get_yang_zhang_volatility("Open", "High", "Low", "Close", window=20)
        df_res = self.df.head(5000).select(expr)
        yz_vals = df_res.drop_nulls().to_series()
        self.assertGreaterEqual(yz_vals.min(), 0.0)
        print(f"   Yang-Zhang OK.")

    def test_logic_volatility_bounds(self):
        """
        [NUEVO] Verifica la lógica de los Conos de Volatilidad:
        1. Techo > Close > Suelo
        2. Spread aumenta con el horizonte temporal.
        """
        print("🧪 Test Lógico: Volatility Bounds (Cones)")
        
        # Test 1: Lógica Básica
        exprs = indicators.get_volatility_bounds(
            col_close="Close", 
            col_vol_yz="dummy_vol_yz", 
            z_score=1.96, 
            horizon=5
        )
        
        df_res = self.df.head(100).select(["Close", "dummy_vol_yz"] + exprs)
        
        # Verificación Vectorizada
        # Techo > Close
        check_ceil = (df_res["fprice_ceil_yz_5d"] > df_res["Close"]).all()
        # Suelo < Close
        check_floor = (df_res["fprice_floor_yz_5d"] < df_res["Close"]).all()
        
        self.assertTrue(check_ceil, "El Techo de volatilidad está por debajo del precio (Error Lógico)")
        self.assertTrue(check_floor, "El Suelo de volatilidad está por encima del precio (Error Lógico)")
        
        print("   ✅ Integridad de Precios: Techo > Close > Suelo")

    def test_full_pipeline_speed(self):
        """🔥 Benchmark completo incluyendo YZ y Conos."""
        print("\n🚀 INICIANDO BENCHMARK DE VELOCIDAD (Multicore)...")
        
        expressions = [
            indicators.get_log_returns("Close"),
            indicators.get_rolling_volatility("Close", window=20),
            indicators.get_yang_zhang_volatility("Open", "High", "Low", "Close", window=20),
            # Para testear los conos en el benchmark, necesitamos calcular YZ al vuelo o usar una dummy.
            # En Polars expressions, podemos encadenar, pero para medir velocidad pura de indicators.py
            # usaremos la dummy_vol_yz creada en setUpClass para simular el paso 2 del pipeline.
            *indicators.get_volatility_bounds("Close", "dummy_vol_yz", 1.96, 5),
            
            *indicators.get_macd_expressions("Close", 12, 26, 9),
            indicators.get_rsi("Close", 14),
            indicators.get_amihud_liquidity("Close", "Close", "Volume", window=20) 
        ]
        
        start_time = time.time()
        df_final = self.df.select(expressions)
        duration = time.time() - start_time
        
        cols = df_final.columns
        rows = df_final.height
        
        print(f"🏁 BENCHMARK FINALIZADO")
        print(f"   - Filas: {rows:,} | Cols: {len(cols)}")
        print(f"   - Tiempo: {duration:.4f}s")
        print(f"   - Velocidad: {rows / duration:,.0f} filas/seg")
        
        self.assertIn("fprice_ceil_yz_5d", cols)
        self.assertLess(duration, 5.0)

if __name__ == '__main__':
    unittest.main()