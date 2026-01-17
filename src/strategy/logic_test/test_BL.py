import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN DEL PATH
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core (Root)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.strategy.motor.black_litterman import BlackLittermanModel

class TestBlackLitterman(unittest.TestCase):

    def setUp(self):
        """Configuración común antes de cada test."""
        # Generamos fechas ficticias para 1 año
        self.dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(252)]
        
    def _create_mock_market_data(self, tickers=["A", "B"], rows=252):
        """Helper para generar datos de mercado falsos rápidamente."""
        data = []
        for t in tickers:
            # Random walk simple
            prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, rows))
            for d, p in zip(self.dates[:rows], prices):
                data.append({"Date": d, "ticker": t, "Close": p})
        
        return pl.DataFrame(data).with_columns(pl.col("Date").cast(pl.Datetime))

    @patch('src.strategy.motor.black_litterman.strat_config')
    @patch('src.strategy.motor.black_litterman.MarketLoader')
    @patch('src.strategy.motor.black_litterman.get_strategy_returns')
    def test_logic_fusion_bayesiana(self, mock_get_strat, mock_loader_cls, mock_config):
        """
        PRUEBA LÓGICA:
        Verifica que el modelo fusiona correctamente el Prior y la View.
        """
        print("\n🧪 [TEST] Lógica Bayesiana...")
        
        # 1. Configuración Mock
        mock_config.TICKERS_ESTRATEGIA = ["ASSET_A", "ASSET_B"]
        mock_config.BLACK_LITTERMAN_CONFIG = {
            "RISK_AVERSION": 2.5,
            "TAU": 0.05,
            "OPTIMIZATION_HORIZON": 5
        }
        mock_config.META_MODEL_CONFIG = {"FORECAST_HORIZON": 5}

        # 2. Datos de Mercado Mock (Polars)
        df_market = self._create_mock_market_data(tickers=["ASSET_A", "ASSET_B"])
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.query.return_value = df_market
        mock_loader_cls.return_value = mock_loader_instance

        # 3. Datos de Estrategia Mock (View fuerte bajista para A)
        strategy_data = {
            "Ticker": ["ASSET_A", "ASSET_B"],
            "Exp_Ret_%": [-5.0, 1.0], 
            "P_Up": [0.10, 0.55],
            "P_Down": [0.90, 0.45]
        }
        mock_get_strat.return_value = pl.DataFrame(strategy_data)

        # 4. Ejecución
        bl = BlackLittermanModel()
        df_result = bl.run_optimization()

        print(df_result)

        # 5. Aserciones
        row_a = df_result.filter(pl.col("Ticker") == "ASSET_A").row(0, named=True)
        
        prior_a = row_a["Prior_Eq_%"]
        view_a = row_a["View_Scaled_%"]
        post_a = row_a["BL_Post_%"]

        self.assertIsNotNone(view_a, "La View no debería ser nula para ASSET_A")
        self.assertLess(post_a, prior_a, 
            f"El BL Posterior ({post_a}) debería haber bajado respecto al Prior ({prior_a}).")
        
        print("✅ Lógica Bayesiana verificada correctamente.")

    @patch('src.strategy.motor.black_litterman.strat_config')
    @patch('src.strategy.motor.black_litterman.MarketLoader')
    @patch('src.strategy.motor.black_litterman.get_strategy_returns')
    def test_missing_views_fallback(self, mock_get_strat, mock_loader_cls, mock_config):
        """
        PRUEBA FALLBACK: Sin views, BL == Prior.
        """
        print("\n🧪 [TEST] Fallback sin Views...")
        
        mock_config.TICKERS_ESTRATEGIA = ["ASSET_A"]
        mock_config.BLACK_LITTERMAN_CONFIG = {"RISK_AVERSION": 2.5, "TAU": 0.05, "OPTIMIZATION_HORIZON": 5}
        mock_config.META_MODEL_CONFIG = {"FORECAST_HORIZON": 5}
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.query.return_value = self._create_mock_market_data(["ASSET_A"])
        mock_loader_cls.return_value = mock_loader_instance
        
        mock_get_strat.return_value = pl.DataFrame([]) 

        bl = BlackLittermanModel()
        df_result = bl.run_optimization()
        
        val_prior = df_result["Prior_Eq_%"][0]
        val_post = df_result["BL_Post_%"][0]
        
        self.assertAlmostEqual(val_prior, val_post, places=5, 
            msg="Sin views, el Posterior debe ser IDÉNTICO al Prior.")
            
        print("✅ Fallback verificado: Prior == Posterior.")

    @patch('src.strategy.motor.black_litterman.strat_config')
    @patch('src.strategy.motor.black_litterman.MarketLoader')
    @patch('src.strategy.motor.black_litterman.get_strategy_returns')
    def test_efficiency_stress(self, mock_get_strat, mock_loader_cls, mock_config):
        """
        PRUEBA DE ESTRÉS: 50 activos, 5 años.
        """
        print("\n🧪 [TEST] Eficiencia Computacional (Stress Test)...")
        
        n_assets = 50
        tickers = [f"TICK_{i}" for i in range(n_assets)]
        
        mock_config.TICKERS_ESTRATEGIA = tickers
        mock_config.BLACK_LITTERMAN_CONFIG = {"RISK_AVERSION": 3.0, "TAU": 0.05, "OPTIMIZATION_HORIZON": 5}
        mock_config.META_MODEL_CONFIG = {"FORECAST_HORIZON": 5}

        # Generar datos masivos
        n_days = 1260
        # Usamos numpy datetime64 directamente para compatibilidad con Polars
        base_date = np.datetime64('2020-01-01')
        dates = base_date + np.arange(n_days)
        
        print(f"   Generando datos sintéticos ({n_assets} activos x {n_days} días)...")
        
        # Repetir fechas y tickers
        all_dates = np.tile(dates, n_assets)
        all_tickers = np.repeat(tickers, n_days)
        all_prices = np.random.uniform(100, 200, size=n_assets*n_days)
        
        # --- FIX FINAL: Polars maneja nativamente numpy datetime64 ---
        df_large = pl.DataFrame({
            "Date": all_dates,
            "ticker": all_tickers,
            "Close": all_prices
        })
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.query.return_value = df_large
        mock_loader_cls.return_value = mock_loader_instance
        
        strat_rows = []
        for t in tickers[:25]:
            strat_rows.append({
                "Ticker": t, "Exp_Ret_%": 2.0, "P_Up": 0.8, "P_Down": 0.2
            })
        mock_get_strat.return_value = pl.DataFrame(strat_rows)

        start_time = time.time()
        
        bl = BlackLittermanModel()
        bl.run_optimization()
        
        duration = time.time() - start_time
        
        print(f"   ⏱️ Tiempo de ejecución: {duration:.4f} segundos")
        self.assertLess(duration, 1.5, "El algoritmo es demasiado lento (>1.5s).")
        
        print("✅ Prueba de eficiencia superada.")

if __name__ == '__main__':
    unittest.main()