import unittest
import time
import numpy as np
import polars as pl
import sys
import os

# --- SETUP DE RUTAS ---
# Truco para importar el módulo indicators que está dos niveles arriba
current_dir = os.path.dirname(os.path.abspath(__file__))
# Subimos a 'src/engine/features'
features_dir = os.path.dirname(current_dir)
# Bajamos a 'src_features'
src_features_path = os.path.join(features_dir, "src_features")

sys.path.append(src_features_path)

try:
    import indicators
except ImportError as e:
    raise ImportError(f"No se pudo importar 'indicators.py'. Revisa la ruta: {src_features_path}") from e

class TestIndicatorsPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Generamos un dataset MASIVO en memoria para probar la velocidad real.
        Simulamos: 500 activos x 5000 días = 2.5 Millones de filas.
        """
        print("\n🌊 Generando Datos Dummy (2.5 Millones de filas)...")
        n_rows = 2_500_000
        
        # Semilla para reproducibilidad
        rng = np.random.default_rng(42)
        
        cls.df = pl.DataFrame({
            "Close": rng.uniform(10, 200, n_rows),
            "Open": rng.uniform(10, 200, n_rows), # No se usa en las formulas actuales pero por si acaso
            "High": rng.uniform(200, 210, n_rows),
            "Low": rng.uniform(5, 10, n_rows),
            "Volume": rng.uniform(1000, 1000000, n_rows),
            # Simulamos retornos calculados previamente o dejamos que la función lo haga
            # Para el test integral, usaremos las columnas base.
        })
        
        # Casting a Float64 para máxima precisión
        cls.df = cls.df.with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Float64)
        ])
        
        print(f"✅ Dataset listo: {cls.df.estimated_size('mb'):.2f} MB en RAM.")

    def test_logic_rsi(self):
        """Verifica que el RSI esté acotado entre 0 y 100."""
        print("\n🧪 Test Lógico: RSI Range")
        expr = indicators.get_rsi("Close", period=14)
        
        # Ejecutamos solo una parte pequeña para validación lógica rápida
        df_res = self.df.head(1000).select(expr)
        
        # Quitamos los nulos del principio (periodo de calentamiento)
        rsi_vals = df_res.drop_nulls().to_series()
        
        min_val = rsi_vals.min()
        max_val = rsi_vals.max()
        
        self.assertGreaterEqual(min_val, 0)
        self.assertLessEqual(max_val, 100)
        print(f"   RSI OK (Min: {min_val:.2f}, Max: {max_val:.2f})")

    def test_logic_efficiency_ratio(self):
        """Verifica que el KER esté entre 0 y 1."""
        print("🧪 Test Lógico: Efficiency Ratio (KER)")
        expr = indicators.get_efficiency_ratio_ker("Close", window=10)
        
        df_res = self.df.head(1000).select(expr)
        ker_vals = df_res.drop_nulls().to_series()
        
        self.assertGreaterEqual(ker_vals.min(), 0.0)
        self.assertLessEqual(ker_vals.max(), 1.0)
        print("   KER OK")

    def test_full_pipeline_speed(self):
        """
        🔥 PRUEBA DE FUEGO: Calcula TODOS los indicadores para 2.5 Millones de filas.
        Aquí es donde Polars debe brillar.
        """
        print("\n🚀 INICIANDO BENCHMARK DE VELOCIDAD (Multicore)...")
        
        # Definimos todas las expresiones a calcular de golpe
        # Esto permite a Polars optimizar el grafo y paralelizar al máximo
        expressions = [
            indicators.get_log_returns("Close"),
            indicators.get_rolling_volatility("Close", window=20), # Nota: Normalmente se haría sobre returns, pero para test de carga sirve Close
            indicators.get_volume_std("Volume", window=20),
            indicators.get_rolling_skewness("Close", window=60),
            indicators.get_volume_return_correlation("Close", "Volume", window=20),
            indicators.get_relative_sma("Close", window=15),
            indicators.get_relative_sma("Close", window=50),
            indicators.get_efficiency_ratio_ker("Close", window=10),
            indicators.get_parkinson_volatility("High", "Low", window=14),
            # Macd devuelve una lista, la desempaquetamos
            *indicators.get_macd_expressions("Close", 12, 26, 9),
            indicators.get_rsi("Close", 14),
            # Amihud requiere retornos absolutos, calculamos al vuelo
            indicators.get_amihud_liquidity("Close", "Close", "Volume", window=20) 
            # Nota: En Amihud he puesto "Close" como proxy de retorno absoluto para el test de carga
            # En producción se pasaría la columna de returns calculada.
        ]
        
        start_time = time.time()
        
        # --- AQUÍ OCURRE LA MAGIA ---
        # Polars ejecuta todo en paralelo aquí
        df_final = self.df.select(expressions).collect() if isinstance(self.df, pl.LazyFrame) else self.df.select(expressions)
        
        end_time = time.time()
        duration = end_time - start_time
        
        cols = df_final.columns
        rows = df_final.height
        
        print(f"🏁 BENCHMARK FINALIZADO")
        print(f"   - Filas procesadas: {rows:,}")
        print(f"   - Indicadores calculados: {len(cols)}")
        print(f"   - Tiempo total: {duration:.4f} segundos")
        print(f"   - Velocidad: {rows / duration:,.0f} filas/segundo")
        
        # Assert de rendimiento: Debería tardar menos de 2 segundos en una CPU moderna
        # (Es un margen muy holgado, Polars suele hacerlo en < 0.5s)
        self.assertLess(duration, 5.0, "El cálculo es demasiado lento (>5s)")

if __name__ == '__main__':
    unittest.main()
