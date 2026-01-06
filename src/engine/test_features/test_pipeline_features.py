import unittest
import os
import sys
import shutil
import time
import tracemalloc  # Para medir memoria
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
        """Configuración: Entorno temporal y Datasets Sintéticos Grandes."""
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
        """Genera un dataset volumétrico para pruebas de estrés."""
        print(f"   Generating mock data: {n_tickers} tickers x {n_days} days...")
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n_days)]
        
        data = []
        sectors = ["Tech", "Banks", "Energy", "Health", "Consum"]
        tickers = [f"TICKER_{i}" for i in range(n_tickers)]
        
        rng = np.random.default_rng(42)
        
        # Vectorizamos la creación de datos para no tardar en el setup
        # (Simulamos brownian motion vectorizado)
        for t in tickers:
            sector = rng.choice(sectors)
            # Random Walk
            returns = rng.normal(0.0005, 0.02, n_days)
            price_path = 100 * np.cumprod(1 + returns)
            
            # Generamos OHLCV coherente
            closes = price_path
            highs = closes * (1 + rng.uniform(0, 0.02, n_days))
            lows = closes * (1 - rng.uniform(0, 0.02, n_days))
            opens = closes * (1 + rng.normal(0, 0.005, n_days)) # Ruido alrededor del close
            vols = rng.integers(10000, 1000000, n_days)
            
            # Construcción eficiente
            tk_data = pl.DataFrame({
                "Date": dates,
                "ticker": [t]*n_days,
                "sector": [sector]*n_days,
                "Close": closes,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Volume": vols,
                "data_quality": [1]*n_days
            })
            data.append(tk_data)
            
        return pl.concat(data)

    @patch('pipeline_features.MarketLoader')
    def test_pipeline_performance_and_integrity(self, MockLoader):
        """
        Benchmark: Mide tiempo, memoria y valida propiedades estadísticas.
        """
        print("\n🔥 INICIANDO STRESS TEST DEL PIPELINE...")
        
        # 1. GENERACIÓN DE CARGA (50 tickers * 500 días = 25,000 filas con lógica compleja)
        # Puedes aumentar n_tickers a 500 para ver la escalabilidad real.
        df_stress = self.generate_stress_data(n_tickers=50, n_days=500)
        
        # Mock del Loader
        mock_instance = MockLoader.return_value
        mock_instance.get_all_data.return_value = df_stress
        
        # Configuración de prueba
        test_params = config.FEATURES_PARAMS.copy()
        test_norm = config.NORMALIZATION_PARAMS.copy()
        test_norm["ROLLING_WINDOW"] = 60  # Ventana más corta para tener datos válidos rápido
        test_norm["MIN_PERIODS"] = 30
        
        # Iniciar medición de recursos
        tracemalloc.start()
        start_time = time.time()
        
        # --- EJECUCIÓN ---
        with patch.dict(config.FEATURES_PARAMS, test_params), \
             patch.dict(config.NORMALIZATION_PARAMS, test_norm), \
             patch.dict(config.PATHS, {"FEATURES_OUTPUT": self.fake_output_path}), \
             patch('pipeline_features.project_root', self.test_dir):
            
            pipeline_features.run_pipeline()

        # --- MÉTRICAS ---
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = end_time - start_time
        peak_mb = peak / 10**6
        
        print(f"   ⏱️  Tiempo de Ejecución: {duration:.4f} segundos")
        print(f"   🧠 Pico de Memoria RAM: {peak_mb:.2f} MB")
        
        # --- VALIDACIONES CIENTÍFICAS ---
        df_res = pl.read_parquet(self.fake_output_path)
        
        # 1. Integridad de Datos
        self.assertFalse(df_res.is_empty())
        self.assertTrue("rsi_14_rob" in df_res.columns)
        self.assertTrue("rsi_14_neutral" in df_res.columns)
        
        # 2. Validación Estadística (Z-Score Robusto debería centrar en 0)
        # Filtramos los nulos iniciales causados por rolling window
        valid_data = df_res.drop_nulls(subset=["rsi_14_neutral"])
        
        if valid_data.height > 0:
            mean_val = valid_data["rsi_14_neutral"].mean()
            std_val = valid_data["rsi_14_neutral"].std()
            
            print(f"   Stats (RSI Neutral): Mean={mean_val:.4f}, Std={std_val:.4f}")
            
            # La media debería estar muy cerca de 0 (ej. +/- 0.5 es aceptable dado el ruido)
            self.assertTrue(-0.5 < mean_val < 0.5, f"La neutralización falló, media desviada: {mean_val}")
            
            # Amihud check (debe ser positivo o cero, nunca negativo si es iliquidez absoluta)
            # Pero como está normalizado (z-score), puede ser negativo.
            # Chequeamos el raw mejor.
            # (Nota: El script guarda todo. Si guardaste raw, verificamos raw).
            
        print("   ✅ Test de Rendimiento y Lógica Matemática SUPERADO.")

if __name__ == '__main__':
    unittest.main()