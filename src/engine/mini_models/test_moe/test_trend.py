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
from datetime import datetime, timedelta

# --- AJUSTE DE PATH PARA IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# Imports del Proyecto
import src.engine.config as config
# Importamos el módulo específico de TREND
from src.engine.mini_models.src_mini_models import trend_mini_model

class TestTrendMiniModel(unittest.TestCase):
    
    def setUp(self):
        """
        Configuración del entorno de prueba.
        """
        self.save_dir = os.path.join(current_dir, "temp_models_trend")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- CONFIGURACIÓN DINÁMICA ---
        # 1. Leemos la ventana de volatilidad real
        self.yz_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
        
        # 2. Leemos las features que el modelo TREND espera encontrar
        # (Esto es lo que diferencia este test del de reversión)
        self.expected_features = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
        
        # 3. Ajustes de Test
        config.MINI_MODEL_PARAMS["FORECAST_HORIZON"] = 5
        config.FEATURES_PARAMS["YZ_Z_SCORE"] = 2.0 
        
        # Forzamos modo producción para que entrene con todo
        config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]["TEST_MODE"] = False

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def _generate_synthetic_data(self, n_rows=250, scenario="random"):
        """
        Genera datos OHLCV sintéticos alineados con FEATURES_TREND.
        """
        np.random.seed(42)
        
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        if scenario == "random":
            # Random Walk
            close = 100 + np.cumsum(np.random.normal(0, 1, n_rows))
            high = close + np.random.uniform(0.1, 2.0, n_rows)
            low = close - np.random.uniform(0.1, 2.0, n_rows)
            vol = np.random.uniform(0.01, 0.05, n_rows) 
            
        elif scenario == "trend_breakout":
            # Escenario de Tendencia Fuerte
            # 1. Creamos una tendencia lineal agresiva
            close = np.linspace(100, 300, n_rows) # Subida fuerte (200 puntos en n_rows)
            # Añadimos un poco de ruido
            close += np.random.normal(0, 0.5, n_rows)
            high = close + 0.5
            low = close - 0.5
            
            # 2. TRUCO CLAVE: Volatilidad MUY BAJA
            # Esto estrecha las bandas de Bollinger/YZ para que el precio las rompa
            # Distancia barrera aprox: Price * 0.001 * 2 = 0.2 puntos.
            # Movimiento diario aprox: 0.2 puntos.
            # Resultado: Rupturas constantes -> Targets = 1 -> Modelo aprende.
            vol = np.full(n_rows, 0.001) 
        
        # 1. Datos Base
        vol_col_name = f"vol_yz_{self.yz_window}d"
        data_dict = {
            "Date": dates,
            "ticker": ["TREND_TEST"] * n_rows,
            "Close": close,
            "High": high,
            "Low": low,
            vol_col_name: vol
        }
        
        # 2. INYECCIÓN DE FEATURES DE TENDENCIA
        for feat in self.expected_features:
            if feat not in data_dict:
                # Inyectamos señal en las features para ayudar al modelo
                if scenario == "trend_breakout":
                    # Si hay tendencia, el ADX suele ser alto y el MACD positivo
                    if "adx" in feat: data_dict[feat] = np.random.normal(30, 5, n_rows)
                    elif "macd" in feat: data_dict[feat] = np.random.normal(1.0, 0.5, n_rows)
                    else: data_dict[feat] = np.random.randn(n_rows)
                else:
                    data_dict[feat] = np.random.randn(n_rows)
        
        return pl.DataFrame(data_dict)
    @patch('src.engine.mini_models.src_mini_models.trend_mini_model.master_features.get_feature_matrix')
    def test_01_pipeline_integrity(self, mock_get_features):
        """
        Verifica que el pipeline carga las features de TENDENCIA y guarda los modelos.
        """
        print("\n--- Test 01: Integridad del Pipeline de Tendencia ---")
        mock_df = self._generate_synthetic_data(n_rows=250, scenario="random")
        mock_get_features.return_value = mock_df
        
        paths = trend_mini_model.train_trend_model("TREND_TEST", self.save_dir)
        
        if paths:
            print("   > Modelos Trend (UP/DOWN) generados correctamente.")
            self.assertTrue(os.path.exists(paths['up']))
            self.assertTrue(os.path.exists(paths['down']))
        else:
            self.fail("El modelo devolvió None (posible fallo de datos insuficientes).")

    @patch('src.engine.mini_models.src_mini_models.trend_mini_model.master_features.get_feature_matrix')
    def test_02_computational_efficiency(self, mock_get_features):
        """
        Stress Test: 10,000 velas con features de tendencia.
        """
        print("\n--- Test 02: Eficiencia Computacional (Trend) ---")
        n_rows = 10000
        mock_df = self._generate_synthetic_data(n_rows=n_rows, scenario="random")
        mock_get_features.return_value = mock_df
        
        start_time = time.time()
        paths = trend_mini_model.train_trend_model("TREND_TEST", self.save_dir)
        duration = time.time() - start_time
        
        print(f"   Tiempo: {duration:.4f}s")
        if duration > 0:
            print(f"   Velocidad: {n_rows / duration:.0f} filas/segundo")
        
        self.assertIsNotNone(paths)
        self.assertLess(duration, 15.0, "El modelo de tendencia es demasiado lento.")

    @patch('src.engine.mini_models.src_mini_models.trend_mini_model.master_features.get_feature_matrix')
    def test_03_learning_capability(self, mock_get_features):
        """
        Verifica que el modelo produce probabilidades variables (no colapsa).
        """
        print("\n--- Test 03: Capacidad de Aprendizaje (No-Collapse) ---")
        mock_df = self._generate_synthetic_data(n_rows=1000, scenario="trend_breakout")
        mock_get_features.return_value = mock_df
        
        paths = trend_mini_model.train_trend_model("TREND_TEST", self.save_dir)
        
        model_up = joblib.load(paths['up'])
        X_test = mock_df.select(self.expected_features).to_pandas()
        
        probs = model_up.predict_proba(X_test)[:, 1]
        
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        
        print(f"   Probabilidad Media UP: {mean_prob:.4f}")
        print(f"   Desviación Estándar:   {std_prob:.4f}")
        
        # Si la std es 0, el modelo devuelve siempre el mismo valor (malo)
        self.assertGreater(std_prob, 0.00001, "El modelo ha colapsado (predicciones estáticas).")

    @patch('src.engine.mini_models.src_mini_models.trend_mini_model.master_features.get_feature_matrix')
    def test_04_trend_visualization(self, mock_get_features):
        """
        Genera gráfico para visualizar las barreras en un contexto de tendencia.
        """
        print("\n--- Test 04: Visualización de Barreras de Tendencia ---")
        mock_df = self._generate_synthetic_data(n_rows=300, scenario="trend_breakout")
        mock_get_features.return_value = mock_df
        
        trend_mini_model.train_trend_model("TREND_TEST", self.save_dir)
        
        # Plotting
        z = config.FEATURES_PARAMS["YZ_Z_SCORE"]
        vol_col = f"vol_yz_{self.yz_window}d"
        
        df = mock_df.clone()
        dates = df["Date"].to_numpy()
        close = df["Close"].to_numpy()
        vol = df[vol_col].to_numpy()
        
        upper = close + (close * vol * z) 
        lower = close - (close * vol * z)
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, close, label='Close Price (Trend)', color='blue', alpha=0.8)
        plt.fill_between(dates, upper, lower, color='cyan', alpha=0.1, label='Volatility Channel')
        
        # Añadimos anotación si es tendencia alcista simulada
        plt.title(f"Trend Model Logic Check (Z={z}) - Scenario: Strong Trend")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(current_dir, "trend_logic_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"   🖼️ Gráfico generado: {output_path}")

if __name__ == '__main__':
    unittest.main()