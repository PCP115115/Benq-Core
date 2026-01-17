import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import polars as pl

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(strategy_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# Importamos el módulo a testear
from src.strategy.motor.sizing import PositionSizer

class TestPositionSizer(unittest.TestCase):

    def setUp(self):
        pass

    @patch('src.strategy.motor.sizing.strat_config')
    @patch('src.strategy.motor.sizing.PortfolioOptimizer')
    def test_sizing_logic_reduction(self, mock_optimizer_cls, mock_config):
        """
        PRUEBA: Reducción de Exposición (De-leveraging).
        Escenario: Cartera muy volátil (30%) vs Target bajo (10%).
        Resultado: Debería invertir solo 1/3 del capital.
        """
        print("\n🧪 [TEST] Lógica de Reducción de Riesgo...")
        
        # 1. Configuración Mock
        mock_config.SIZING_CONFIG = {
            "TOTAL_CAPITAL": 100_000.0,
            "TARGET_VOLATILITY_ANNUAL": 0.10, # Target 10%
            "MAX_LEVERAGE": 1.5,
            "MIN_CASH_BUFFER": 0.0
        }
        # Horizonte optimización: 252 días (para simplificar cálculo anual)
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 252}

        # 2. Mock Optimizer
        mock_opt_instance = MagicMock()
        mock_optimizer_cls.return_value = mock_opt_instance
        
        # Cartera Base: 100% en Activo A
        df_optimal = pl.DataFrame({"Ticker": ["A"], "Weight_%": [100.0]})
        mock_opt_instance.optimize_portfolio.return_value = df_optimal
        
        # Inputs Crudos: Mu, Sigma
        tickers = ["A"]
        mu = np.array([0.05])
        # Sigma = 0.30^2 = 0.09 (Volatilidad 30% anual)
        sigma = np.array([[0.09]]) 
        mock_opt_instance._get_optimization_inputs.return_value = (tickers, mu, sigma)

        # 3. Ejecución
        sizer = PositionSizer()
        df_sizing = sizer.get_final_allocations()
        
        print(df_sizing)

        # 4. Aserciones
        # Vol Cartera = 30%. Target = 10%. Scalar = 1/3 = 0.3333
        row_a = df_sizing.filter(pl.col("Ticker") == "A").row(0, named=True)
        row_cash = df_sizing.filter(pl.col("Ticker") == "CASH (USD)").row(0, named=True)
        
        self.assertAlmostEqual(row_a["Vol_Adj_Weight_%"], 33.33, delta=0.5, 
                               msg="La exposición debería reducirse al ~33.3%")
        self.assertAlmostEqual(row_cash["Vol_Adj_Weight_%"], 66.67, delta=0.5,
                               msg="El cash debería ser el ~66.7%")
        
        print("✅ Reducción correcta.")

    @patch('src.strategy.motor.sizing.strat_config')
    @patch('src.strategy.motor.sizing.PortfolioOptimizer')
    def test_sizing_logic_leverage_cap(self, mock_optimizer_cls, mock_config):
        """
        PRUEBA: Apalancamiento y Cap (Max Leverage).
        Escenario: Cartera muy segura (5%) vs Target (15%).
        Teórico: 3.0x. Límite Configurado: 1.5x.
        Resultado: Debería topar en 1.5x.
        """
        print("\n🧪 [TEST] Lógica de Apalancamiento Máximo...")
        
        mock_config.SIZING_CONFIG = {
            "TOTAL_CAPITAL": 100_000.0,
            "TARGET_VOLATILITY_ANNUAL": 0.15, # Target 15%
            "MAX_LEVERAGE": 1.5,              # Tope 1.5x
            "MIN_CASH_BUFFER": 0.0
        }
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 252}

        mock_opt_instance = MagicMock()
        mock_optimizer_cls.return_value = mock_opt_instance
        
        df_optimal = pl.DataFrame({"Ticker": ["SAFE_ASSET"], "Weight_%": [100.0]})
        mock_opt_instance.optimize_portfolio.return_value = df_optimal
        
        tickers = ["SAFE_ASSET"]
        mu = np.array([0.02])
        # Sigma = 0.05^2 = 0.0025 (Volatilidad 5% anual)
        sigma = np.array([[0.0025]]) 
        mock_opt_instance._get_optimization_inputs.return_value = (tickers, mu, sigma)

        sizer = PositionSizer()
        df_sizing = sizer.get_final_allocations()
        
        print(df_sizing)
        
        row_asset = df_sizing.filter(pl.col("Ticker") == "SAFE_ASSET").row(0, named=True)
        
        # Debería ser 150% (Max Leverage), no 300% (Scalar Teórico)
        self.assertAlmostEqual(row_asset["Vol_Adj_Weight_%"], 150.0, delta=0.5,
                               msg="El apalancamiento debería estar limitado a 150%")
        
        print("✅ Leverage Cap respetado.")

    @patch('src.strategy.motor.sizing.strat_config')
    @patch('src.strategy.motor.sizing.PortfolioOptimizer')
    def test_cash_buffer_integrity(self, mock_optimizer_cls, mock_config):
        """
        PRUEBA: Integridad del Cash Buffer.
        Verifica que siempre se respete el buffer de efectivo mínimo.
        """
        print("\n🧪 [TEST] Integridad de Cash Buffer...")
        
        mock_config.SIZING_CONFIG = {
            "TOTAL_CAPITAL": 1000.0,
            "TARGET_VOLATILITY_ANNUAL": 0.50, # Target Altísimo (Pide apalancamiento infinito)
            "MAX_LEVERAGE": 5.0,
            "MIN_CASH_BUFFER": 0.10 # 10% Cash obligatorio
        }
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 252}

        mock_opt_instance = MagicMock()
        mock_optimizer_cls.return_value = mock_opt_instance
        
        df_optimal = pl.DataFrame({"Ticker": ["A"], "Weight_%": [100.0]})
        mock_opt_instance.optimize_portfolio.return_value = df_optimal
        
        mock_opt_instance._get_optimization_inputs.return_value = (["A"], np.array([0.1]), np.array([[0.01]]))

        sizer = PositionSizer()
        df_sizing = sizer.get_final_allocations()
        
        # Max Leverage Efectivo = 5.0 * (1 - 0.10) = 4.5
        # Exposición esperada = 450%
        
        row_asset = df_sizing.filter(pl.col("Ticker") == "A").row(0, named=True)
        self.assertAlmostEqual(row_asset["Vol_Adj_Weight_%"], 450.0, delta=1.0)
        
        print("✅ Cash Buffer respetado en el cálculo de leverage efectivo.")

if __name__ == '__main__':
    unittest.main()