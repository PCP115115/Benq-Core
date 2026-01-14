import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import pandas as pd
import numpy as np
import sys
import os
import shutil
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# --- AJUSTE DE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# Imports
import src.engine.config as config
from src.engine.mini_models.src_mini_models import master_mini_models

class TestMasterMiniModels(unittest.TestCase):
    
    def setUp(self):
        """
        Setup: Directorios temporales para simular el sistema de archivos.
        """
        self.temp_meta_input_dir = os.path.join(current_dir, "temp_meta_inputs")
        self.temp_models_dir = os.path.join(current_dir, "temp_models_master")
        
        os.makedirs(self.temp_meta_input_dir, exist_ok=True)
        os.makedirs(self.temp_models_dir, exist_ok=True)
        
        # Sobrescribimos las rutas en el módulo master para que apunten a temp
        master_mini_models.META_INPUT_DIR = Path(self.temp_meta_input_dir)
        master_mini_models.MODELS_DIR = Path(self.temp_models_dir)
        
        # Parámetros básicos para que la lógica de split funcione
        config.MINI_MODEL_PARAMS["FORECAST_HORIZON"] = 5
        config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]["TRAIN_TEST_SPLIT_RATIO"] = 0.8
        
        # Recuperamos las features REALES que espera el código para generarlas en el dummy
        self.trend_feats = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
        self.rev_feats = config.MINI_MODEL_PARAMS["FEATURES_REVERSION"]
        self.vol_feats = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.temp_meta_input_dir):
            shutil.rmtree(self.temp_meta_input_dir)
        if os.path.exists(self.temp_models_dir):
            shutil.rmtree(self.temp_models_dir)

    def _create_dummy_data(self, n_rows=200):
        """
        Crea un DF Polars compatible con la configuración real.
        Genera todas las columnas que config.py espera encontrar.
        """
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        # 1. Datos base obligatorios
        data_dict = {
            "Date": dates,
            "ticker": ["TEST"] * n_rows,
            "Close": np.linspace(100, 200, n_rows),
            "log_returns": np.random.normal(0, 0.01, n_rows)
        }
        
        # 2. Inyectamos Features Reales (Trend, Rev, Vol)
        # Combinamos todas las listas y creamos columnas de ruido para cada una
        all_features = set(self.trend_feats + self.rev_feats + self.vol_feats)
        
        for feat in all_features:
            data_dict[feat] = np.random.randn(n_rows)
            
        return pl.DataFrame(data_dict)

    @patch('src.engine.mini_models.src_mini_models.master_mini_models.joblib.load')
    @patch('src.engine.mini_models.src_mini_models.master_mini_models.volatility_mini_models.train_volatility_model')
    @patch('src.engine.mini_models.src_mini_models.master_mini_models.reversion_mini_models.train_reversion_model')
    @patch('src.engine.mini_models.src_mini_models.master_mini_models.trend_mini_model.train_trend_model')
    @patch('src.engine.mini_models.src_mini_models.master_mini_models.master_features.get_feature_matrix')
    def test_01_worker_process_flow(self, mock_get_features, mock_trend, mock_rev, mock_vol, mock_joblib_load):
        """
        Prueba la función 'process_ticker_pipeline' (el Worker).
        """
        print("\n--- Test 01: Flujo Completo del Worker (Orquestación) ---")
        
        # 1. Mock Data (Ahora tiene las columnas correctas adx_14, etc.)
        n_rows = 200
        mock_df = self._create_dummy_data(n_rows)
        mock_get_features.return_value = mock_df
        
        # 2. Mock Training Returns
        mock_trend.return_value = {"up": "path/trend_up", "down": "path/trend_down"}
        mock_rev.return_value = {"up": "path/rev_up", "down": "path/rev_down"}
        mock_vol.return_value = {"expansion": "path/vol_exp", "compression": "path/vol_comp"}
        
        # 3. Mock Model Predictions
        mock_model_instance = MagicMock()
        split_ratio = 0.8
        cutoff = int(n_rows * split_ratio)
        test_rows = n_rows - cutoff
        
        dummy_probs = np.random.rand(test_rows, 2)
        mock_model_instance.predict_proba.return_value = dummy_probs
        
        mock_joblib_load.return_value = mock_model_instance
        
        # 4. EJECUCIÓN DEL WORKER
        result_msg = master_mini_models.process_ticker_pipeline("TEST_TICKER")
        
        print(f"   Resultado del Worker: {result_msg}")
        
        # 5. VALIDACIONES
        self.assertIn("✅", result_msg, "El worker falló. Revisa el log de error arriba.")
        
        expected_parquet = os.path.join(self.temp_meta_input_dir, "meta_input_TEST_TICKER.parquet")
        self.assertTrue(os.path.exists(expected_parquet), "No se generó el archivo de salida Parquet.")
        
        df_result = pl.read_parquet(expected_parquet)
        
        # Validar columnas críticas
        expected_cols = ["P_Trend_Up", "P_Rev_Down", "P_Vol_Exp"]
        for col in expected_cols:
            self.assertIn(col, df_result.columns)

    @patch('src.engine.mini_models.src_mini_models.master_mini_models.as_completed') # <--- CLAVE PARA EVITAR HANG
    @patch('src.engine.mini_models.src_mini_models.master_mini_models.ProcessPoolExecutor')
    def test_02_orchestrator_concurrency(self, mock_executor_cls, mock_as_completed):
        """
        Prueba 'run_mini_models_pipeline'.
        CORREGIDO: Mockeamos as_completed para que no espere infinitamente.
        """
        print("\n--- Test 02: Orquestador Paralelo (Mocked & Non-Blocking) ---")
        
        tickers = ["AAPL", "MSFT", "GOOG"]
        
        # 1. Configurar el Mock del Executor
        mock_executor_instance = mock_executor_cls.return_value.__enter__.return_value
        
        # Configuramos submit para que devuelva un objeto "Future" falso
        mock_future = MagicMock()
        mock_future.result.return_value = "✅ Mock Result OK"
        mock_executor_instance.submit.return_value = mock_future
        
        # 2. Configurar as_completed (EL TRUCO ANTI-BLOQUEO)
        # Hacemos que devuelva inmediatamente la lista de futuros que se le pasaron
        # as_completed recibe un dict {future: ticker}, queremos iterar sobre los futures (keys)
        mock_as_completed.side_effect = lambda futures_dict: list(futures_dict.keys())
        
        # 3. Ejecutar
        master_mini_models.run_mini_models_pipeline(tickers, n_jobs=2)
        
        # 4. Validar
        print(f"   Tareas enviadas al Executor: {mock_executor_instance.submit.call_count}")
        self.assertEqual(mock_executor_instance.submit.call_count, 3, "No se enviaron los 3 tickers.")

if __name__ == '__main__':
    unittest.main()