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

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.dirname(current_dir)      
engine_dir = os.path.dirname(features_dir)       
src_dir = os.path.dirname(engine_dir)            

sys.path.append(features_dir)
sys.path.append(src_dir)
sys.path.append(os.path.join(features_dir, "src_features"))

import pipeline_features
import config

class TestFeaturePipelineHardcore(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(current_dir, "temp_pipeline_bench")
        os.makedirs(self.test_dir, exist_ok=True)
        self.fake_output_path = os.path.join(self.test_dir, "bench_features.parquet")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

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
        
        # Test config (nos aseguramos que los parámetros coincidan con lo esperado)
        test_params = config.FEATURES_PARAMS.copy()
        test_params["YZ_FORECAST_HORIZON"] = 5 # Forzamos 5 para el assert
        
        tracemalloc.start()
        start_time = time.time()
        
        with patch.dict(config.FEATURES_PARAMS, test_params), \
             patch.dict(config.PATHS, {"FEATURES_OUTPUT": self.fake_output_path}), \
             patch('pipeline_features.project_root', self.test_dir):
            
            pipeline_features.run_pipeline()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"   ⏱️  Tiempo: {time.time() - start_time:.4f}s")
        print(f"   🧠 Memoria: {peak / 10**6:.2f} MB")
        
        # --- VALIDACIONES ---
        df_res = pl.read_parquet(self.fake_output_path)
        cols = df_res.columns
        
        # 1. ¿Existen las columnas de los conos?
        col_ceil = "fprice_ceil_yz_5d"
        col_floor = "fprice_floor_yz_5d"
        
        self.assertIn(col_ceil, cols, "Falta columna Techo YZ")
        self.assertIn(col_floor, cols, "Falta columna Suelo YZ")
        
        # 2. Validación Lógica en Pipeline Real
        # Tomamos una muestra aleatoria válida (donde no sea null por el periodo de calentamiento)
        valid_rows = df_res.drop_nulls(subset=[col_ceil]).head(10)
        
        if valid_rows.height > 0:
            ceil_check = (valid_rows[col_ceil] > valid_rows["Close"]).all()
            floor_check = (valid_rows[col_floor] < valid_rows["Close"]).all()
            
            self.assertTrue(ceil_check, "Pipeline generó Techos inválidos (< Close)")
            self.assertTrue(floor_check, "Pipeline generó Suelos inválidos (> Close)")
            print(f"   ✅ Validación lógica Techo/Suelo en datos procesados correcta.")

        # 3. ¿Se normalizaron (Robust Scaler)?
        # Deben existir versiones _rob y _neutral de los precios teóricos
        self.assertIn(f"{col_ceil}_rob", cols)
        self.assertIn(f"{col_ceil}_neutral", cols)
        
        print("   ✅ Test End-to-End con Conos de Volatilidad SUPERADO.")

if __name__ == '__main__':
    unittest.main()