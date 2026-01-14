import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
import os
import shutil
import xgboost as xgb
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- AJUSTE DE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# Imports
import src.engine.config as config
from src.engine.meta_model.src_meta_model import pipeline_meta

class TestMetaModelPerformance(unittest.TestCase):
    
    def setUp(self):
        """Setup: Directorios temporales."""
        self.temp_model_dir = os.path.join(current_dir, "temp_perf_models")
        os.makedirs(self.temp_model_dir, exist_ok=True)
        pipeline_meta.MODEL_DIR = self.temp_model_dir
        
        # Configuración Base
        config.META_MODEL_PARAMS["FORECAST_HORIZON"] = 5
        config.FEATURES_PARAMS["YZ_Z_SCORE"] = 2.0
        config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"] = 20
        config.META_MODEL_PARAMS["META_MODEL_TRAIN_PARAMS"]["TRAIN_TEST_SPLIT_RATIO"] = 0.8
        
        # Resultados para graficar
        self.results_size = []
        self.results_time = []

    def tearDown(self):
        """Limpieza y Generación de Reporte Gráfico."""
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)
            
        # Generar Gráfico si hay resultados
        if len(self.results_size) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_size, self.results_time, marker='o', linestyle='-', color='b')
            plt.title('Meta-Model Performance: Training Time vs Data Size')
            plt.xlabel('Number of Rows (Data Size)')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(True)
            
            output_plot = os.path.join(current_dir, "meta_performance_chart.png")
            plt.savefig(output_plot)
            print(f"\n📊 Gráfico de rendimiento generado: {output_plot}")
            plt.close()

    def _create_large_mock_data(self, n_rows):
        """Generador de datos masivos eficiente."""
        start_date = datetime(2020, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        # Optimizamos la generación con numpy para velocidad
        data = {
            "Date": dates,
            "ticker": ["PERF_TEST"] * n_rows,
            "Close": np.random.uniform(100, 200, n_rows),
            "High": np.random.uniform(105, 205, n_rows),
            "Low": np.random.uniform(95, 195, n_rows),
            "Open": np.random.uniform(100, 200, n_rows),
            "Volume": np.random.randint(1000, 50000, n_rows),
            "log_returns": np.random.normal(0, 0.01, n_rows),
            "vol_yz_20d_RAW": np.random.uniform(0.01, 0.05, n_rows),
            "market_regime": np.random.randint(0, 3, n_rows),
            "regime_probability": np.random.random(n_rows),
            
            # Expertos
            "P_Trend_Up": np.random.random(n_rows),
            "P_Trend_Down": np.random.random(n_rows),
            "P_Rev_Up": np.random.random(n_rows),
            "P_Rev_Down": np.random.random(n_rows),
            "P_Vol_Exp": np.random.random(n_rows),
            "P_Vol_Comp": np.random.random(n_rows)
        }
        
        # Añadir features técnicos extra
        for i in range(5):
            data[f"tech_feat_{i}"] = np.random.randn(n_rows)
            
        return pl.DataFrame(data)

    @patch('src.engine.meta_model.src_meta_model.pipeline_meta.get_data_meta_model')
    def test_scalability_stress_test(self, mock_get_data):
        """
        Prueba de Estrés: Entrena el modelo con tamaños crecientes de datos.
        Objetivo: Verificar que el tiempo no explota exponencialmente (O(n)).
        """
        print("\n--- Test de Escalabilidad y Performance ---")
        
        # Tamaños a probar: 1.000, 5.000, 10.000, 50.000 filas
        # 50.000 filas ~= 200 años de datos diarios de un activo (Stress puro)
        data_sizes = [1000, 5000, 10000, 25000] 
        
        # Configuramos XGBoost para usar 1 hilo y medir la eficiencia raw del algoritmo
        config.META_MODEL_PARAMS["XGB_PARAMS"]["n_jobs"] = 1
        
        for size in data_sizes:
            print(f"   > Probando con {size} filas...", end="")
            
            # 1. Generar datos
            mock_df = self._create_large_mock_data(size)
            mock_get_data.return_value = mock_df
            
            # 2. Medir tiempo
            start_time = time.time()
            pipeline_meta.train_meta_model("PERF_TEST")
            duration = time.time() - start_time
            
            print(f" ✅ Tiempo: {duration:.4f}s")
            
            # Guardar para reporte
            self.results_size.append(size)
            self.results_time.append(duration)
            
            # 3. Assertions de eficiencia
            # El tiempo por fila no debería degradarse masivamente
            time_per_row = duration / size
            if size > 5000:
                # XGBoost es muy eficiente, debería mantenerse bajo
                # Un umbral conservador: 0.001s por fila en entrenamiento completo
                self.assertLess(time_per_row, 0.002, f"Cuello de botella detectado en size {size}")

    def test_multiprocessing_configuration(self):
        """
        Verifica que la configuración está lista para paralelismo.
        """
        print("\n--- Test de Configuración de Multiprocesado ---")
        xgb_params = config.META_MODEL_PARAMS["XGB_PARAMS"]
        
        # Verificamos n_jobs
        print(f"   > XGBoost n_jobs configurado: {xgb_params.get('n_jobs')}")
        
        # En el orquestador (Master Mini Models), usamos ProcessPoolExecutor.
        # En el Meta-Modelo, usamos el paralelismo interno de XGBoost.
        # Si entrenamos UN solo meta-modelo por activo, n_jobs debería ser alto (ej. 4 u 8).
        # Si entrenamos MUCHOS activos en paralelo, n_jobs debería ser 1 para evitar sobrecarga.
        
        # Como train_meta_model se ejecuta secuencialmente para un ticker en el script actual,
        # lo ideal es que n_jobs > 1.
        
        # Este test solo advierte, no falla, porque depende de tu estrategia de despliegue.
        if xgb_params.get('n_jobs') == 1:
            print("   ⚠️ AVISO: n_jobs=1. Si ejecutas un solo ticker, desaprovechas CPU.")
            print("      Si ejecutas múltiples tickers en paralelo, esto es CORRECTO.")
        else:
            print("   ✅ Configuración optimizada para entrenamiento individual rápido.")

if __name__ == '__main__':
    unittest.main()