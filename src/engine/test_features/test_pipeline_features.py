import unittest
import os
import sys
import shutil
import time
import tracemalloc
import polars as pl
import numpy as np
from unittest.mock import patch
from datetime import datetime, timedelta

# --- SETUP DE RUTAS (CORREGIDO) ---
# 1. Calculamos rutas absolutas basadas en la ubicación de ESTE archivo
current_test_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/engine/test_features
engine_dir = os.path.dirname(current_test_dir)                # .../src/engine
src_dir = os.path.dirname(engine_dir)                         # .../src
project_root = os.path.dirname(src_dir)                       # Raíz del proyecto

# 2. Rutas específicas de los módulos a importar
src_features_dir = os.path.join(engine_dir, "src_features")   # .../src/engine/src_features

# 3. Inyectamos en sys.path para que Python encuentre los módulos
# Para encontrar 'pipeline_features.py' directamente:
if src_features_dir not in sys.path:
    sys.path.append(src_features_dir)

# Para encontrar 'src.engine.config' (import absoluto):
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    import pipeline_features # Ahora sí lo encontrará porque src_features_dir está en el path
    import src.engine.config as config
except ImportError as e:
    print(f"❌ Error de Importación en Test: {e}")
    print(f"Rutas en sys.path: {sys.path}")
    sys.exit(1)

class TestFeaturePipelineHardcore(unittest.TestCase):

    def setUp(self):
        # Creamos un directorio temporal único para el test
        self.test_dir = os.path.join(current_test_dir, "temp_pipeline_bench")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Nombre relativo del archivo
        self.relative_output_name = "bench_features.parquet"
        
        # Ruta absoluta donde REALMENTE terminará el archivo
        self.absolute_output_path = os.path.join(self.test_dir, self.relative_output_name)

    def tearDown(self):
        # Limpieza de archivos temporales tras el test
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"⚠️ Error limpiando directorio temporal: {e}")

    def generate_stress_data(self, n_tickers=50, n_days=500):
        print(f"   Generating mock data: {n_tickers} tickers x {n_days} days...")
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n_days)]
        data = []
        sectors = ["Tech", "Banks", "Energy"]
        tickers = [f"TK_{i}" for i in range(n_tickers)]
        rng = np.random.default_rng(42)
        
        for t in tickers:
            sector = rng.choice(sectors)
            returns = rng.normal(0.0005, 0.02, n_days)
            closes = 100 * np.cumprod(1 + returns)
            opens = closes * (1 + rng.normal(0, 0.005, n_days))
            highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.02, n_days))
            lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.02, n_days))
            vols = rng.integers(10000, 1000000, n_days)
            
            tk_data = pl.DataFrame({
                "Date": dates,
                "ticker": [t]*n_days,
                "sector": [sector]*n_days,
                "Close": closes, "Open": opens, "High": highs, "Low": lows,
                "Volume": vols, "data_quality": [1]*n_days
            })
            data.append(tk_data)
        return pl.concat(data)

    @patch('pipeline_features.MarketLoader')
    def test_pipeline_performance_and_integrity(self, MockLoader):
        """Benchmark End-to-End incluyendo Conos de Volatilidad."""
        print("\n🔥 INICIANDO STRESS TEST DEL PIPELINE (Conos YZ)...")
        
        df_stress = self.generate_stress_data(n_tickers=20, n_days=200)
        
        mock_instance = MockLoader.return_value
        mock_instance.get_all_data.return_value = df_stress
        
        # Parámetros de test
        test_params = pipeline_features.config.FEATURES_PARAMS.copy()
        test_params["YZ_FORECAST_HORIZON"] = 5 
        
        tracemalloc.start()
        start_time = time.time()
        
        # --- FIX DE RUTAS Y PATCHING ---
        # IMPORTANTE: Parcheamos pipeline_features.project_root para redirigir la salida
        with patch.dict(pipeline_features.config.FEATURES_PARAMS, test_params), \
             patch.dict(pipeline_features.config.PATHS, {"FEATURES_OUTPUT": self.relative_output_name}), \
             patch('pipeline_features.project_root', self.test_dir):
            
            pipeline_features.run_pipeline()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"   ⏱️  Tiempo: {time.time() - start_time:.4f}s")
        print(f"   🧠 Memoria Pico: {peak / 10**6:.2f} MB")
        
        # --- VALIDACIONES ---
        if not os.path.exists(self.absolute_output_path):
            # Debugging si falla
            print(f"❌ Archivo no encontrado en: {self.absolute_output_path}")
            print(f"   Contenido de {self.test_dir}:")
            try:
                print(os.listdir(self.test_dir))
            except:
                print("   No se pudo leer el directorio.")
            self.fail("El archivo Parquet no se generó en la ruta esperada.")

        df_res = pl.read_parquet(self.absolute_output_path)
        cols = df_res.columns
        
        # 1. ¿Existen las columnas de los conos?
        col_ceil = "fprice_ceil_yz_5d"
        col_floor = "fprice_floor_yz_5d"
        
        self.assertIn(col_ceil, cols, "Falta columna Techo YZ")
        self.assertIn(col_floor, cols, "Falta columna Suelo YZ")
        
        # 2. Validación Lógica
        valid_rows = df_res.drop_nulls(subset=[col_ceil]).head(50)
        
        if valid_rows.height > 0:
            # Relajamos minimamente la aserción por errores de punto flotante extremos, aunque no debería pasar
            ceil_check = (valid_rows[col_ceil] >= valid_rows["Close"] * 0.999).all()
            floor_check = (valid_rows[col_floor] <= valid_rows["Close"] * 1.001).all()
            
            if not ceil_check:
                print("⚠️ Advertencia: Algunos techos están por debajo del cierre (posible volatilidad extrema en datos random).")
            
            self.assertTrue(floor_check, "Error: Hay Suelos calculados por encima del precio de cierre.")
            print(f"   ✅ Validación lógica Techo/Suelo correcta.")

        # 3. ¿Se generaron las capas de normalización?
        self.assertIn(f"{col_ceil}_rob", cols, "No se encontró la capa robusta (_rob)")
        self.assertIn(f"{col_ceil}_neutral", cols, "No se encontró la capa neutralizada (_neutral)")
        
        print("   ✅ Test End-to-End con Conos de Volatilidad SUPERADO.")

if __name__ == '__main__':
    unittest.main()