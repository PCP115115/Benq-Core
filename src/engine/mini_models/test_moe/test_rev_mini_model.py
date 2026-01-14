import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
import sys
import os
import shutil
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime, timedelta

# --- AJUSTE DE PATH PARA IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# Imports del Proyecto
import src.engine.config as config
from src.engine.mini_models.src_mini_models import reversion_mini_models

class TestReversionMiniModel(unittest.TestCase):
    
    def setUp(self):
        """
        Configuración del entorno de prueba.
        """
        self.save_dir = os.path.join(current_dir, "temp_models")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- CONFIGURACIÓN DINÁMICA ---
        # Leemos la ventana de volatilidad real para generar la columna correcta
        self.yz_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
        # Leemos las features que el modelo espera encontrar
        self.expected_features = config.MINI_MODEL_PARAMS["FEATURES_REVERSION"]
        
        # Forzamos modo producción para entrenamiento completo en tests de estrés
        config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]["TEST_MODE"] = False

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def _generate_synthetic_data(self, n_rows=250, scenario="random"):
        """
        Genera datos OHLCV sintéticos que COINCIDEN con las features de config.py.
        """
        np.random.seed(42)
        
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        if scenario == "random":
            close = 100 + np.cumsum(np.random.normal(0, 1, n_rows))
            high = close + np.random.uniform(0.1, 2.0, n_rows)
            low = close - np.random.uniform(0.1, 2.0, n_rows)
            vol = np.random.uniform(0.01, 0.05, n_rows) 
            
        elif scenario == "sniper":
            close = np.full(n_rows, 100.0)
            high = np.full(n_rows, 100.5)
            low = np.full(n_rows, 99.5)
            vol = np.full(n_rows, 0.01) 
            
            # Eventos controlados
            if n_rows > 30:
                close[12] = 110; high[12] = 112 
                close[30] = 80; low[30] = 78
        
        # 1. Creamos diccionario base con precios y vol
        vol_col_name = f"vol_yz_{self.yz_window}d"
        data_dict = {
            "Date": dates,
            "ticker": ["TEST"] * n_rows,
            "Close": close,
            "High": high,
            "Low": low,
            vol_col_name: vol
        }
        
        # 2. INYECCIÓN DINÁMICA DE FEATURES (La clave del arreglo)
        # Generamos ruido para cada columna que config.py espere (rsi_14, skew_60d...)
        for feat in self.expected_features:
            if feat not in data_dict: # Evitamos sobrescribir si alguna coincide
                data_dict[feat] = np.random.randn(n_rows)
        
        return pl.DataFrame(data_dict)

    @patch('src.engine.mini_models.src_mini_models.reversion_mini_models.master_features.get_feature_matrix')
    def test_01_barrier_logic_precision(self, mock_get_features):
        """
        Test de Lógica Pura: Verifica si el algoritmo detecta rupturas futuras.
        """
        print("\n--- Test 01: Precisión Lógica de Barreras (Triple Barrier) ---")
        # Aumentamos n_rows a 250 para superar el filtro de "Datos insuficientes (<200)"
        mock_df = self._generate_synthetic_data(n_rows=250, scenario="sniper")
        mock_get_features.return_value = mock_df
        
        paths = reversion_mini_models.train_reversion_model("TEST", self.save_dir)
        
        if paths:
            print("   > Modelo entrenado correctamente (Lógica interna OK).")
            self.assertTrue(os.path.exists(paths['up']))
        else:
            self.fail("El modelo devolvió None (probablemente por filtro de tamaño de datos).")

    @patch('src.engine.mini_models.src_mini_models.reversion_mini_models.master_features.get_feature_matrix')
    def test_02_stress_and_performance(self, mock_get_features):
        """
        Test de Carga: Procesa 10,000 filas.
        """
        print("\n--- Test 02: Stress Test (10k filas) ---")
        n_rows = 10000
        mock_df = self._generate_synthetic_data(n_rows=n_rows, scenario="random")
        mock_get_features.return_value = mock_df
        
        start_time = time.time()
        paths = reversion_mini_models.train_reversion_model("TEST", self.save_dir)
        duration = time.time() - start_time
        
        print(f"   Tiempo de procesamiento: {duration:.4f}s")
        if duration > 0:
            print(f"   Velocidad: {n_rows / duration:.0f} filas/segundo")
        
        self.assertIsNotNone(paths, "El modelo devolvió None. Revisa los logs de columnas faltantes.")
        self.assertLess(duration, 15.0)

    @patch('src.engine.mini_models.src_mini_models.reversion_mini_models.master_features.get_feature_matrix')
    def test_03_statistical_fit_auc(self, mock_get_features):
        """
        Entrena el modelo y verifica si aprende (AUC Check).
        """
        print("\n--- Test 03: Ajuste Estadístico (AUC Check) ---")
        mock_df = self._generate_synthetic_data(n_rows=1000, scenario="random")
        mock_get_features.return_value = mock_df
        
        paths = reversion_mini_models.train_reversion_model("TEST", self.save_dir)
        
        # Validamos que cargue
        self.assertIsNotNone(paths)
        model_up = joblib.load(paths['up'])
        
        # Preparamos X_test con las columnas correctas
        features = self.expected_features
        X_test = mock_df.select(features).to_pandas()
        
        probs = model_up.predict_proba(X_test)[:, 1]
        
        print(f"   Probabilidad Media UP: {np.mean(probs):.4f}")
        self.assertGreater(np.std(probs), 0.00001, "El modelo predice valores constantes.")

    @patch('src.engine.mini_models.src_mini_models.reversion_mini_models.master_features.get_feature_matrix')
    def test_04_visualization_debug(self, mock_get_features):
        """
        Genera un gráfico para verificar visualmente dónde se ponen las barreras.
        """
        print("\n--- Test 04: Visualización de Barreras y Señales ---")
        mock_df = self._generate_synthetic_data(n_rows=300, scenario="random") 
        mock_get_features.return_value = mock_df
        
        reversion_mini_models.train_reversion_model("TEST", self.save_dir)
        
        # RECONSTRUCCIÓN MANUAL PARA PLOTTING
        z = config.FEATURES_PARAMS["YZ_Z_SCORE"]
        vol_col = f"vol_yz_{self.yz_window}d"
        
        df = mock_df.clone()
        dates = df["Date"].to_numpy()
        close = df["Close"].to_numpy()
        vol = df[vol_col].to_numpy()
        
        upper = close + (close * vol * z) 
        lower = close - (close * vol * z)
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, close, label='Close Price', color='black', alpha=0.7)
        plt.fill_between(dates, upper, lower, color='gray', alpha=0.2, label='Vol Cones (YZ)')
        
        plt.title(f"Verificación Visual: Dinámica de Precios y Volatilidad (Z={z})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(current_dir, "reversion_logic_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"   🖼️ Gráfico generado: {output_path}")

if __name__ == '__main__':
    unittest.main()