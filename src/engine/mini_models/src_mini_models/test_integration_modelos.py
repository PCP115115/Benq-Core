import unittest
import os
import sys
import shutil
import time
import numpy as np
import polars as pl
import joblib
from unittest.mock import patch, MagicMock

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mini_models_dir = os.path.dirname(current_dir)       
engine_dir = os.path.dirname(mini_models_dir)        
src_dir = os.path.dirname(engine_dir)                

sys.path.append(src_dir)
sys.path.append(engine_dir)
sys.path.append(os.path.join(engine_dir, "src_features"))
sys.path.append(os.path.join(mini_models_dir, "src_mini_models"))

import config
import master_mini_models
from trend_mini_model import train_trend_model
from volatility_mini_models import train_volatility_model
from reversion_mini_models import train_reversion_model

class TestMiniModelsIntegration(unittest.TestCase):
    
    def setUp(self):
        """Preparación del entorno."""
        self.test_dir = os.path.join(current_dir, "test_output_temp")
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except:
                pass
        os.makedirs(self.test_dir, exist_ok=True)
        self.features_parquet = os.path.join(self.test_dir, "features_matrix_test.parquet")
        
    def tearDown(self):
        """Limpieza robusta para Windows."""
        # Esperamos un poco a que el multiprocessing suelte los archivos
        time.sleep(1.0)
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"\n⚠️ Aviso: Windows bloqueó el borrado de {self.test_dir}. Bórralo manualmente si quieres.")
            except Exception as e:
                print(f"\n⚠️ Error limpiando test: {e}")

    def generate_dummy_data_exact(self, n_tickers=2, n_rows=200):
        """Genera datos sintéticos basados en tu config real."""
        dfs = []
        from datetime import datetime, timedelta
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_rows)]
        
        f_trend = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
        f_vol = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
        f_rev = config.MINI_MODEL_PARAMS["FEATURES_REVERSION"]
        all_features = list(set(f_trend + f_vol + f_rev))
        
        for i in range(n_tickers):
            ticker_name = f"TICKER_{i:02d}"
            
            base_data = {
                "Date": dates,
                "ticker": [ticker_name] * n_rows,
                "Open": np.random.uniform(100, 200, n_rows),
                "High": np.random.uniform(100, 200, n_rows),
                "Low": np.random.uniform(100, 200, n_rows),
                "Close": np.random.uniform(100, 200, n_rows),
                "Volume": np.random.uniform(1000, 50000, n_rows),
            }
            
            base_data["High"] = np.maximum(base_data["Open"], base_data["Close"]) + 2.0
            base_data["Low"] = np.minimum(base_data["Open"], base_data["Close"]) - 2.0
            
            for feat in all_features:
                base_data[feat] = np.random.uniform(10, 50, n_rows)
                base_data[f"{feat}_rob"] = np.random.uniform(0, 1, n_rows)
            
            horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
            base_data[f"fprice_ceil_yz_{horizon}d"] = np.random.uniform(210, 220, n_rows)
            base_data[f"fprice_floor_yz_{horizon}d"] = np.random.uniform(80, 90, n_rows)
            
            yz_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
            base_data[f"vol_yz_{yz_window}d"] = np.random.uniform(0.01, 0.05, n_rows)
            
            dfs.append(pl.DataFrame(base_data))
            
        return pl.concat(dfs)

    def test_01_training_logic_individual(self):
        print("\n🧪 TEST 01: Lógica Individual de Entrenamiento...")
        df_dummy = self.generate_dummy_data_exact(n_tickers=1, n_rows=150)
        ticker = "TICKER_00"
        
        # Trend
        path_trend = train_trend_model(df_dummy, ticker, self.test_dir)
        self.assertTrue(os.path.exists(path_trend))
        
        # Volatility
        path_vol = train_volatility_model(df_dummy, ticker, self.test_dir)
        self.assertTrue(os.path.exists(path_vol))
        
        # Reversion
        path_rev = train_reversion_model(df_dummy, ticker, self.test_dir)
        self.assertTrue(os.path.exists(path_rev))
        print("   ✅ Modelos individuales generados correctamente.")

    def test_02_orchestrator_integration(self):
        """
        🔥 PRUEBA INTEGRAL CORREGIDA
        """
        print("\n🧪 TEST 02: Orquestador Completo (Multiprocessing)...")
        
        N_TICKERS = 2
        df_dummy = self.generate_dummy_data_exact(n_tickers=N_TICKERS, n_rows=200)
        
        # Guardar parquet físico
        df_dummy.write_parquet(self.features_parquet)
        
        # Configurar rutas absolutas para el patch
        new_paths = config.PATHS.copy()
        new_paths["FEATURES_OUTPUT"] = os.path.abspath(self.features_parquet)
        # Importante: Añadir separador al final para que os.path.join funcione bien
        new_paths["MINI_MODELS_DIR"] = os.path.abspath(self.test_dir) + os.sep
        
        with patch.dict(config.PATHS, new_paths):
            with patch("master_mini_models.master_features.get_feature_matrix") as mock_get_matrix:
                
                mock_get_matrix.return_value = df_dummy.lazy().collect() 
                
                start_time = time.time()
                master_mini_models.train_all_models()
                duration = time.time() - start_time
                print(f"⏱️  Duración: {duration:.4f}s")
                
                # --- VALIDACIÓN CORREGIDA (RECURSIVA) ---
                # Tus modelos se guardan en subcarpetas (TICKER_XX/trend.joblib)
                # así que os.listdir() en la raíz no los ve. Usamos os.walk.
                found_models = []
                for root, dirs, files in os.walk(self.test_dir):
                    for file in files:
                        if file.endswith(".joblib"):
                            found_models.append(os.path.join(root, file))
                
                print(f"   📂 Archivos encontrados: {len(found_models)}")
                
                # Verificamos cantidad: 2 tickers * 3 modelos = 6
                self.assertEqual(len(found_models), N_TICKERS * 3, 
                                 f"Se esperaban {N_TICKERS*3} modelos, se encontraron {len(found_models)}")
                
                print("   ✅ Orquestador finalizó correctamente.")

if __name__ == '__main__':
    unittest.main(verbosity=2)