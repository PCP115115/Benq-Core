import unittest
from unittest.mock import MagicMock, patch
import polars as pl
from datetime import timedelta
import os
import sys
import warnings

# --- 1. CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 2. MOCKING DE DEPENDENCIAS EXTERNAS ---
sys.modules["tickers"] = MagicMock()
# Mockeamos loader completo antes de importar features
sys.modules["loader"] = MagicMock()

from features import FeatureEngine

class TestFeatureEngine(unittest.TestCase):

    def setUp(self):
        """Configuración previa a cada test."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Mock de parámetros
        self.mock_params = {
            "RSI_PERIOD": 2, "VOLATILITY_WINDOW": 3, "TREND_WINDOW": 3,
            "REL_VOL_WINDOW": 3, "MACD_FAST": 2, "MACD_SLOW": 5, "MACD_SIGNAL": 2
        }
        
        with patch("features.FEATURES_PARAMS", self.mock_params):
            # IMPORTANTE: Ahora el FeatureEngine instancia un MarketLoader real dentro.
            # Debemos interceptar esa creación.
            with patch("features.MarketLoader") as MockLoaderClass:
                # Instanciamos el motor
                self.engine = FeatureEngine()
                self.engine.params = self.mock_params
                
                # Accedemos a la instancia mockeada del loader que creó el engine
                self.mock_loader_instance = self.engine.loader

    def create_dummy_dataframe(self):
        """
        Crea un DataFrame (EAGER) simulando lo que devuelve DuckDB.
        """
        N_ROWS = 50 
        
        # Ticker A: Tendencia alcista
        data_a = {
            "Date": pl.date_range(start=pl.date(2023, 1, 1), end=pl.date(2023, 1, 1).map_elements(lambda x: x + timedelta(days=N_ROWS-1), return_dtype=pl.Date), eager=True),
            "ticker": ["A"] * N_ROWS,
            "sector": ["Tech"] * N_ROWS,
            "Open": [10.0] * N_ROWS, "High": [10.0] * N_ROWS, "Low": [10.0] * N_ROWS,
            "Close": [float(i * 10) for i in range(1, N_ROWS + 1)], 
            "Volume": [100] * N_ROWS,
            "data_quality": [1] * N_ROWS 
        }
        
        # Ticker B: Precio plano
        data_b = {
            "Date": pl.date_range(start=pl.date(2023, 1, 1), end=pl.date(2023, 1, 1).map_elements(lambda x: x + timedelta(days=N_ROWS-1), return_dtype=pl.Date), eager=True),
            "ticker": ["B"] * N_ROWS,
            "sector": ["Energy"] * N_ROWS,
            "Open": [100.0] * N_ROWS, "High": [100.0] * N_ROWS, "Low": [100.0] * N_ROWS,
            "Close": [100.0] * N_ROWS, 
            "Volume": [500] * N_ROWS,
            "data_quality": [1] * N_ROWS
        }
        
        df = pl.DataFrame(data_a).vstack(pl.DataFrame(data_b))
        
        # Casting estricto como vendría de DuckDB/Loader
        return df.with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Int64),
            pl.col("data_quality").cast(pl.Int8)
        ])

    def test_new_indicators_existence(self):
        """Valida que el pipeline procese los datos entregados por el mock loader."""
        
        # 1. Configurar el Mock
        # Cuando engine llame a loader.get_all_data(), devuelve nuestro DF falso
        dummy_df = self.create_dummy_dataframe()
        self.mock_loader_instance.get_all_data.return_value = dummy_df
        
        # 2. Ejecutar
        df_result = self.engine.get_market_dataset()
        
        # 3. Validar
        cols = df_result.columns
        self.assertIn("EWNA", cols)
        self.assertIn("Rolling_Z_Score", cols)
        self.assertIn("Efficiency_Ratio_KER", cols)
        
        # Verificar que hay datos
        self.assertTrue(df_result.height > 0)

    def test_ker_logic(self):
        """Prueba lógica matemática."""
        # Crear dato perfecto para KER=1
        N = 30
        data_efficient = {
            "Date": pl.date_range(start=pl.date(2023, 1, 1), end=pl.date(2023, 1, 1).map_elements(lambda x: x + timedelta(days=N-1), return_dtype=pl.Date), eager=True),
            "ticker": ["EFF"] * N,
            "sector": ["Test"] * N,
            "Open": [10.0]*N, "High": [10.0]*N, "Low": [10.0]*N,
            "Close": [float(i) for i in range(10, 10 + N)],
            "Volume": [100]*N,
            "data_quality": [1]*N
        }
        df_eff = pl.DataFrame(data_efficient).with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Int64),
            pl.col("data_quality").cast(pl.Int8)
        ])

        # Mockear respuesta del loader
        self.mock_loader_instance.get_all_data.return_value = df_eff
        
        df_result = self.engine.get_market_dataset()
        
        ker_vals = df_result["Efficiency_Ratio_KER"].tail(3).to_list()
        for v in ker_vals:
            self.assertAlmostEqual(v, 1.0, places=1)

    def test_pipeline_integration_via_sql_filter(self):
        """
        Simula que pedimos un sector específico y verificamos que el engine
        llama a loader.query con el filtro SQL adecuado.
        """
        dummy_df = self.create_dummy_dataframe()
        self.mock_loader_instance.query.return_value = dummy_df
        
        # Ejecutar pidiendo sector 'Tech'
        self.engine.get_market_dataset(sector="Tech")
        
        # Verificar que features.py llamó a loader.query con SQL
        # Esto confirma que la delegación a DuckDB está ocurriendo
        self.mock_loader_instance.query.assert_called()
        args, _ = self.mock_loader_instance.query.call_args
        sql_query = args[0]
        
        self.assertIn("SELECT * FROM market", sql_query)
        self.assertIn("sector = 'Tech'", sql_query)

if __name__ == "__main__":
    unittest.main(verbosity=2)