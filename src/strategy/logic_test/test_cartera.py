import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch
import polars as pl
import numpy as np

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(strategy_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# Importamos el módulo bajo prueba para asegurar que está cargado
import src.strategy.motor.cartera as cartera_module
from src.strategy.motor.cartera import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        pass

    # Usamos patch sobre el módulo importado para asegurar la ruta correcta
    @patch('src.strategy.motor.cartera.strat_config')
    @patch('src.strategy.motor.cartera.BlackLittermanModel')
    @patch('src.strategy.motor.cartera.MarketLoader')
    def test_optimization_logic_max_sharpe(self, mock_loader, mock_bl_cls, mock_config):
        """PRUEBA LÓGICA: MAX_SHARPE"""
        print("\n🧪 [TEST] Lógica de Optimización (Max Sharpe)...")
        
        mock_config.TICKERS_ESTRATEGIA = ["A", "B", "C"]
        mock_config.PORTFOLIO_CONFIG = {
            "OBJECTIVE": "MAX_SHARPE",
            "RISK_FREE_RATE_ANNUAL": 0.0,
            "ALLOW_SHORTS": False,
            "MAX_WEIGHT_PER_ASSET": 1.0,
            "MIN_WEIGHT_PER_ASSET": 0.0
        }
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 5}

        # Mock BL
        mock_bl_instance = MagicMock()
        mock_bl_cls.return_value = mock_bl_instance
        
        # Retornos: A gana, B medio, C pierde
        df_bl = pl.DataFrame({
            "Ticker": ["A", "B", "C"],
            "BL_Post_%": [5.0, 2.0, -1.0] 
        })
        mock_bl_instance.run_optimization.return_value = df_bl
        mock_bl_instance._get_historical_returns.return_value = (MagicMock(), ["A", "B", "C"])
        mock_bl_instance._estimate_covariance_scaled.return_value = np.eye(3) * 0.01

        optimizer = PortfolioOptimizer()
        df_res = optimizer.optimize_portfolio()
        
        print(df_res)
        weights = {row["Ticker"]: row["Weight_%"] for row in df_res.to_dicts()}
        
        self.assertAlmostEqual(sum(weights.values()), 100.0, delta=0.1)
        self.assertGreater(weights["A"], weights["B"])
        self.assertEqual(weights["C"], 0.0)
        print("✅ Lógica Sharpe verificada.")

    @patch('src.strategy.motor.cartera.strat_config')
    @patch('src.strategy.motor.cartera.BlackLittermanModel')
    @patch('src.strategy.motor.cartera.MarketLoader')
    def test_constraints_max_weight(self, mock_loader, mock_bl_cls, mock_config):
        """PRUEBA DE RESTRICCIONES: MAX WEIGHT"""
        print("\n🧪 [TEST] Restricción de Peso Máximo (40%)...")
        
        mock_config.TICKERS_ESTRATEGIA = ["A", "B", "C"]
        mock_config.PORTFOLIO_CONFIG = {
            "OBJECTIVE": "MAX_SHARPE",
            "RISK_FREE_RATE_ANNUAL": 0.0,
            "ALLOW_SHORTS": False,
            "MAX_WEIGHT_PER_ASSET": 0.40,
            "MIN_WEIGHT_PER_ASSET": 0.0
        }
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 5}

        mock_bl_instance = MagicMock()
        mock_bl_cls.return_value = mock_bl_instance
        
        # Activo A muy atractivo (debería topar en 40%)
        df_bl = pl.DataFrame({
            "Ticker": ["A", "B", "C"],
            "BL_Post_%": [10.0, 0.1, 0.1] 
        })
        mock_bl_instance.run_optimization.return_value = df_bl
        mock_bl_instance._get_historical_returns.return_value = (MagicMock(), ["A", "B", "C"])
        mock_bl_instance._estimate_covariance_scaled.return_value = np.eye(3) * 0.01

        optimizer = PortfolioOptimizer()
        df_res = optimizer.optimize_portfolio()
        
        print(df_res)
        weights = {row["Ticker"]: row["Weight_%"] for row in df_res.to_dicts()}
        
        self.assertAlmostEqual(weights["A"], 40.0, delta=0.1)
        print("✅ Restricciones de peso respetadas.")

    @patch('src.strategy.motor.cartera.strat_config')
    @patch('src.strategy.motor.cartera.BlackLittermanModel')
    @patch('src.strategy.motor.cartera.MarketLoader')
    def test_efficiency_large_portfolio(self, mock_loader, mock_bl_cls, mock_config):
        """PRUEBA DE ESTRÉS: 50 Activos"""
        print("\n🧪 [TEST] Eficiencia Computacional (50 Activos)...")
        
        n_assets = 50
        tickers = [f"T_{i}" for i in range(n_assets)]
        
        mock_config.TICKERS_ESTRATEGIA = tickers
        mock_config.PORTFOLIO_CONFIG = {
            "OBJECTIVE": "MAX_SHARPE",
            "RISK_FREE_RATE_ANNUAL": 0.04,
            "ALLOW_SHORTS": False,
            "MAX_WEIGHT_PER_ASSET": 1.0,
            "MIN_WEIGHT_PER_ASSET": 0.0
        }
        mock_config.BLACK_LITTERMAN_CONFIG = {"OPTIMIZATION_HORIZON": 5}

        mock_bl_instance = MagicMock()
        mock_bl_cls.return_value = mock_bl_instance
        
        mu = np.random.uniform(-0.01, 0.02, n_assets)
        df_bl = pl.DataFrame({"Ticker": tickers, "BL_Post_%": mu * 100})
        mock_bl_instance.run_optimization.return_value = df_bl
        mock_bl_instance._get_historical_returns.return_value = (MagicMock(), tickers)
        
        A = np.random.rand(n_assets, n_assets)
        sigma = np.dot(A, A.T)
        mock_bl_instance._estimate_covariance_scaled.return_value = sigma

        start_time = time.time()
        optimizer = PortfolioOptimizer()
        optimizer.optimize_portfolio()
        duration = time.time() - start_time
        
        print(f"   ⏱️ Tiempo de ejecución: {duration:.4f} segundos")
        self.assertLess(duration, 0.5)
        print("✅ Prueba de eficiencia superada.")

if __name__ == '__main__':
    unittest.main()