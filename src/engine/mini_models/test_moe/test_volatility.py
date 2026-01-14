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
from src.engine.mini_models.src_mini_models import volatility_mini_models

class TestVolatilityMiniModel(unittest.TestCase):
    
    def setUp(self):
        """
        Configuración del entorno de prueba.
        """
        self.save_dir = os.path.join(current_dir, "temp_models_vol")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- CONFIGURACIÓN DINÁMICA (ANTI-FALLOS) ---
        # 1. Leemos la ventana YZ real para generar la columna correcta
        self.yz_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
        
        # 2. Leemos las features que el modelo VOLATILITY espera
        self.expected_features = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
        
        # 3. Ajustes de Test
        config.MINI_MODEL_PARAMS["FORECAST_HORIZON"] = 5
        config.FEATURES_PARAMS["YZ_Z_SCORE"] = 2.0 
        
        # Forzamos modo producción para entrenamiento completo
        config.MINI_MODEL_PARAMS["MINI_MODEL_TRAIN_PARAMS"]["TEST_MODE"] = False

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def _generate_synthetic_data(self, n_rows=250, scenario="random"):
        """
        Genera datos sintéticos con log_returns y features dinámicas.
        """
        np.random.seed(42)
        
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        # --- GENERACIÓN DE PRECIOS Y RETORNOS ---
        if scenario == "random":
            # Mercado normal
            returns = np.random.normal(0, 0.01, n_rows) # 1% volatilidad diaria
        elif scenario == "vol_explosion":
            # Mercado que se vuelve loco al final
            # Primera mitad: Calma (0.5%). Segunda mitad: Pánico (3%).
            n_calm = n_rows // 2
            n_panic = n_rows - n_calm
            returns = np.concatenate([
                np.random.normal(0, 0.005, n_calm),
                np.random.normal(0, 0.03, n_panic)
            ])
            
        # Reconstruimos precios desde retornos (Price = 100 * exp(cumsum(ret)))
        close = 100 * np.exp(np.cumsum(returns))
        # High/Low simulados
        high = close * 1.01
        low = close * 0.99
        
        # Columna clave: log_returns (El script la exige)
        log_returns = returns
        
        # Columna clave: Volatilidad Actual (YZ o Std)
        # Simulamos que nuestra estimación YZ es una media móvil de la volatilidad real reciente
        vol_col_name = f"vol_yz_{self.yz_window}d"
        # Hacemos una aproximación simple de la vol rolling para rellenar
        vol_yz_sim = pl.Series(returns).rolling_std(window_size=self.yz_window).fill_null(0.01).to_numpy()
        
        # 1. Diccionario Base
        data_dict = {
            "Date": dates,
            "ticker": ["VOL_TEST"] * n_rows,
            "Close": close,
            "High": high,
            "Low": low,
            "log_returns": log_returns, # <--- CRÍTICO
            vol_col_name: vol_yz_sim
        }
        
        # 2. INYECCIÓN DINÁMICA DE FEATURES EXTRA
        for feat in self.expected_features:
            if feat not in data_dict:
                data_dict[feat] = np.random.randn(n_rows)
        
        return pl.DataFrame(data_dict)

    @patch('src.engine.mini_models.src_mini_models.volatility_mini_models.master_features.get_feature_matrix')
    def test_01_pipeline_integrity(self, mock_get_features):
        """
        Verifica que el pipeline carga features y calcula targets sin romper.
        """
        print("\n--- Test 01: Integridad del Pipeline de Volatilidad ---")
        mock_df = self._generate_synthetic_data(n_rows=250, scenario="random")
        mock_get_features.return_value = mock_df
        
        paths = volatility_mini_models.train_volatility_model("VOL_TEST", self.save_dir)
        
        if paths:
            print("   > Modelos Volatility (EXP/COM) generados correctamente.")
            self.assertTrue(os.path.exists(paths['expansion']))
            self.assertTrue(os.path.exists(paths['compression']))
        else:
            self.fail("El modelo devolvió None (posible fallo de datos o missing columns).")

    @patch('src.engine.mini_models.src_mini_models.volatility_mini_models.master_features.get_feature_matrix')
    def test_02_computational_efficiency(self, mock_get_features):
        """
        Stress Test: 10,000 velas con cálculo de targets rolling.
        """
        print("\n--- Test 02: Eficiencia Computacional (Volatility) ---")
        n_rows = 10000
        mock_df = self._generate_synthetic_data(n_rows=n_rows, scenario="random")
        mock_get_features.return_value = mock_df
        
        start_time = time.time()
        paths = volatility_mini_models.train_volatility_model("VOL_TEST", self.save_dir)
        duration = time.time() - start_time
        
        print(f"   Tiempo: {duration:.4f}s")
        if duration > 0:
            print(f"   Velocidad: {n_rows / duration:.0f} filas/segundo")
        
        self.assertIsNotNone(paths)
        self.assertLess(duration, 15.0, "El modelo de volatilidad es demasiado lento.")

    @patch('src.engine.mini_models.src_mini_models.volatility_mini_models.master_features.get_feature_matrix')
    def test_03_learning_capability(self, mock_get_features):
        """
        Verifica que el modelo detecta la expansión de volatilidad.
        """
        print("\n--- Test 03: Capacidad de Aprendizaje (Vol Expansion) ---")
        # Usamos escenario de explosión para que haya señales claras
        mock_df = self._generate_synthetic_data(n_rows=1000, scenario="vol_explosion")
        mock_get_features.return_value = mock_df
        
        paths = volatility_mini_models.train_volatility_model("VOL_TEST", self.save_dir)
        
        model_exp = joblib.load(paths['expansion'])
        X_test = mock_df.select(self.expected_features).to_pandas()
        
        probs = model_exp.predict_proba(X_test)[:, 1]
        
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        
        print(f"   Probabilidad Media Expansion: {mean_prob:.4f}")
        print(f"   Desviación Estándar:        {std_prob:.4f}")
        
        self.assertGreater(std_prob, 0.00001, "El modelo ha colapsado (predicciones estáticas).")

    @patch('src.engine.mini_models.src_mini_models.volatility_mini_models.master_features.get_feature_matrix')
    def test_04_volatility_target_debug(self, mock_get_features):
        """
        Genera gráfico para validar visualmente la lógica del Target.
        Compara Volatilidad Actual vs Volatilidad Futura Realizada.
        """
        print("\n--- Test 04: Visualización de Targets de Volatilidad ---")
        mock_df = self._generate_synthetic_data(n_rows=300, scenario="vol_explosion")
        mock_get_features.return_value = mock_df
        
        volatility_mini_models.train_volatility_model("VOL_TEST", self.save_dir)
        
        # RECONSTRUCCIÓN LÓGICA DEL TARGET PARA PLOT
        horizon = config.MINI_MODEL_PARAMS["FORECAST_HORIZON"]
        vol_col_name = f"vol_yz_{self.yz_window}d"
        
        df = mock_df.clone()
        
        # 1. Volatilidad Actual (Lo que sabe el modelo hoy)
        current_vol = df[vol_col_name].to_numpy()
        
        # 2. Volatilidad Futura (Lo que intenta predecir - La Verdad)
        # rolling_std shiftado hacia atrás (-horizon)
        future_vol = (
            df["log_returns"]
            .rolling_std(window_size=horizon)
            .shift(-horizon)
            .to_numpy()
        )
        
        dates = np.arange(len(df))
        
        plt.figure(figsize=(12, 6))
        
        # Plot Volatilidades
        plt.plot(dates, current_vol, label='Current Vol (YZ Estimate)', color='blue', alpha=0.6)
        plt.plot(dates, future_vol, label=f'Future Realized Vol ({horizon}d)', color='red', linestyle='--', alpha=0.8)
        
        # Rellenar zonas donde Futuro > Actual (Expansion Target = 1)
        # Usamos where para filtrar nulos por el shift
        valid_mask = ~np.isnan(future_vol)
        plt.fill_between(dates[valid_mask], current_vol[valid_mask], future_vol[valid_mask], 
                         where=(future_vol[valid_mask] > current_vol[valid_mask]),
                         color='red', alpha=0.1, label='Target: Expansion Zone')
        
        plt.title(f"Volatility Logic Check (Horizon={horizon}d)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(current_dir, "volatility_logic_plot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"   🖼️ Gráfico generado: {output_path}")

if __name__ == '__main__':
    unittest.main()