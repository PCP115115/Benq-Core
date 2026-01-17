import sys
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import polars as pl

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(strategy_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.strategy.motor.master_motor import StrategyMasterEngine

class TestStrategyMaster(unittest.TestCase):

    def setUp(self):
        """Crea directorio temporal para tests."""
        self.test_orders_dir = os.path.join(strategy_dir, "test_orders_temp")
        os.makedirs(self.test_orders_dir, exist_ok=True)

    def tearDown(self):
        """Borra directorio temporal."""
        if os.path.exists(self.test_orders_dir):
            shutil.rmtree(self.test_orders_dir)

    @patch('src.strategy.motor.master_motor.strat_config')
    @patch('src.strategy.motor.master_motor.PositionSizer')
    def test_end_to_end_flow_and_filtering(self, mock_sizer_cls, mock_config):
        """Valida que el Master reciba datos, filtre y guarde archivos."""
        print("\n🧪 [TEST] Flujo Maestro y Filtros de Órdenes...")
        
        # Configuración Mock
        mock_config.EXECUTION_CONFIG = {
            "MIN_ORDER_VALUE": 100.0,
            "CHECK_MARKET_OPEN": False,
            "ORDERS_DIR": "test_orders_temp"
        }
        
        # Mock Datos del Sizer (CORREGIDO: Añadida columna Exp_Ret_Horizon_%)
        df_sizing_mock = pl.DataFrame([
            {
                "Ticker": "AAPL", "Role": "LONG", "Capital_Alloc": 5000.0, 
                "Vol_Adj_Weight_%": 50.0, "Exp_Ret_Horizon_%": 1.5
            },
            {
                "Ticker": "PENNY", "Role": "LONG", "Capital_Alloc": 50.0, 
                "Vol_Adj_Weight_%": 0.5, "Exp_Ret_Horizon_%": 0.1
            },
            {
                "Ticker": "CASH (USD)", "Role": "LIQUIDITY", "Capital_Alloc": 4950.0, 
                "Vol_Adj_Weight_%": 49.5, "Exp_Ret_Horizon_%": 0.0
            }
        ])
        
        mock_sizer_instance = MagicMock()
        mock_sizer_instance.get_final_allocations.return_value = df_sizing_mock
        mock_sizer_cls.return_value = mock_sizer_instance

        # Ejecución
        engine = StrategyMasterEngine()
        engine.orders_path = self.test_orders_dir # Forzar ruta de test
        
        df_orders = engine.run()
        
        print(df_orders)

        # Aserciones
        self.assertEqual(df_orders.height, 1, "Solo AAPL debería pasar los filtros.")
        self.assertEqual(df_orders["Ticker"][0], "AAPL")
        
        # Verificar Ficheros
        files = os.listdir(self.test_orders_dir)
        self.assertTrue(any("orders_" in f for f in files), "Debe existir el CSV de órdenes")
        self.assertTrue(any("portfolio_status_" in f for f in files), "Debe existir el CSV de portfolio")
        
        print("✅ Flujo verificado.")

    @patch('src.strategy.motor.master_motor.strat_config')
    @patch('src.strategy.motor.master_motor.PositionSizer')
    @patch('src.strategy.motor.master_motor.datetime')
    def test_market_closed_logic(self, mock_datetime, mock_sizer_cls, mock_config):
        """Valida que el motor se detenga en fin de semana."""
        print("\n🧪 [TEST] Control de Mercado Cerrado...")
        
        mock_config.EXECUTION_CONFIG = {
            "MIN_ORDER_VALUE": 100.0,
            "CHECK_MARKET_OPEN": True,
            "ORDERS_DIR": "test_orders_temp"
        }
        
        # Simular Domingo
        mock_datetime.now.return_value = datetime(2024, 1, 14, 12, 0, 0)
        
        engine = StrategyMasterEngine()
        result = engine.run()
        
        # Aserciones
        self.assertIsNone(result, "El motor debió devolver None")
        mock_sizer_cls.return_value.get_final_allocations.assert_not_called()
        
        print("✅ Detención correcta.")

if __name__ == '__main__':
    unittest.main()