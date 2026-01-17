import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import polars as pl
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DEL PATH PARA IMPORTAR MÓDULOS
# ==========================================
# Permitimos que el test encuentre la raíz del proyecto dinámicamente
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core (Root)

if project_root not in sys.path:
    sys.path.append(project_root)

# Importamos la clase a testear
from src.strategy.motor.returns import MetaStrategyEngine

class TestMetaStrategyEngine(unittest.TestCase):
    
    def setUp(self):
        """Configuración previa a cada test"""
        pass

    # Mockeamos todas las dependencias externas para aislar la lógica
    @patch('src.strategy.motor.returns.strat_config')
    @patch('src.strategy.motor.returns.get_data_meta_model')
    @patch('src.strategy.motor.returns.indicators')
    @patch('src.strategy.motor.returns.xgb.XGBClassifier')
    @patch('os.path.exists')
    def test_calculation_logic(self, mock_exists, mock_xgb, mock_indicators, mock_get_data, mock_config):
        """
        Verifica que el motor calcula correctamente E[R] dados unos inputs controlados.
        Formula: E[R] = (P_up * R_TP) - (P_down * |R_SL|)
        """
        
        # ---------------------------------------------------------
        # 1. CONFIGURACIÓN DEL ESCENARIO (MOCKS)
        # ---------------------------------------------------------
        
        # A. Simulamos Configuración
        mock_config.TICKERS_ESTRATEGIA = ["TEST_ASSET"]
        mock_config.META_MODEL_CONFIG = {
            "FORCE_RETRAIN": False,
            "FORECAST_HORIZON": 4,   # T=4 (Raíz cuadrada = 2, facilita cálculo mental)
            "YZ_Z_SCORE": 2.0,       # Z=2
            "VOL_WINDOW": 20,
            "MIN_PROB_THRESHOLD": 0.50
        }
        mock_config.OUTPUT_CONFIG = {"EXPORT_TO_CSV": False}

        # B. Simulamos que los archivos de modelos existen
        mock_exists.return_value = True

        # C. Simulamos el Modelo XGBoost
        mock_booster = MagicMock()
        # El modelo "pide" estas features (Lista Blanca)
        mock_booster.feature_names = ["rsi_14", "vol_yz_20d"]
        
        mock_model_instance = MagicMock()
        mock_model_instance.get_booster.return_value = mock_booster
        
        # Simulamos predicciones:
        # - Modelo UP dice: 80% probabilidad (Clase 1)
        # - Modelo DOWN dice: 30% probabilidad (Clase 1)
        # predict_proba devuelve [[prob_0, prob_1]]
        mock_model_instance.predict_proba.side_effect = [
            np.array([[0.2, 0.80]]), # Llamada 1 (UP)
            np.array([[0.7, 0.30]])  # Llamada 2 (DOWN)
        ]
        mock_xgb.return_value = mock_model_instance

        # D. Simulamos Datos de Mercado (DataFrame Polars)
        data = {
            "Date": [datetime(2025, 1, 1)],
            "ticker": ["TEST_ASSET"],
            "Close": [100.0],
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "rsi_14": [50.0], # Feature requerida
            # Nota: 'vol_yz_20d' se genera dinámicamente, no hace falta ponerla aquí si mockeamos indicators
        }
        df_mock = pl.DataFrame(data)
        mock_get_data.return_value = df_mock

        # E. Simulamos el Indicador de Volatilidad
        # Hacemos que get_yang_zhang_volatility devuelva un valor fijo de 0.05 (5%)
        # Usamos pl.lit() para que sea una expresión válida de Polars
        mock_indicators.get_yang_zhang_volatility.return_value = pl.lit(0.05).alias("vol_yz_20d")

        # ---------------------------------------------------------
        # 2. EJECUCIÓN DEL MOTOR
        # ---------------------------------------------------------
        
        engine = MetaStrategyEngine()
        engine.initialize()
        df_results = engine.calculate_expected_returns()

        # ---------------------------------------------------------
        # 3. VALIDACIÓN DE RESULTADOS (MATH CHECK)
        # ---------------------------------------------------------
        
        print("\n--- DATOS DE LA PRUEBA ---")
        print(f"Precio: 100 | Vol: 5% | Z: 2 | T: 4 dias")
        print(f"Prob UP: 80% | Prob DOWN: 30%")
        
        # CÁLCULO ESPERADO:
        # Proyección = Vol * Z * sqrt(T)
        #            = 0.05 * 2.0 * 2.0 = 0.20 (20%)
        #
        # R_TP = 20% | R_SL = 20%
        #
        # E[R] = (P_up * R_TP) - (P_down * R_SL)
        #      = (0.80 * 0.20) - (0.30 * 0.20)
        #      = 0.16 - 0.06
        #      = 0.10 (10%)
        
        print("\n--- RESULTADO DEL MOTOR ---")
        print(df_results)
        
        self.assertFalse(df_results.is_empty(), "El DataFrame no debería estar vacío")
        
        # Extraemos el valor calculado por el script
        calc_return = df_results["Exp_Ret_%"][0]
        
        # Validamos con precisión de decimales
        self.assertAlmostEqual(calc_return, 10.0, delta=0.01, 
                               msg=f"El retorno esperado debería ser 10.0%, pero fue {calc_return}%")
        
        # Validamos que NO existan columnas prohibidas (como Signal)
        self.assertNotIn("Signal", df_results.columns, "La columna 'Signal' debería haber sido eliminada")
        
        print("\n✅ ¡PRUEBA EXITOSA! La lógica matemática es correcta.")

if __name__ == '__main__':
    unittest.main()