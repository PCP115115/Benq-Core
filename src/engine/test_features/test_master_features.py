import unittest
import os
import sys
import shutil
import polars as pl
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# --- SETUP DE RUTAS ROBUSTO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Asumimos estructura: src/engine/test_features/test_master.py -> src/engine
engine_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(engine_dir)

sys.path.append(src_dir)
sys.path.append(engine_dir)
# Para encontrar master_features y pipeline_features
sys.path.append(os.path.join(engine_dir, "src_features"))

import master_features

class TestMasterFeaturesPro(unittest.TestCase):

    def setUp(self):
        """Preparamos un entorno aislado con datos falsos."""
        self.test_dir = os.path.join(current_dir, "temp_master_test")
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.fake_parquet_path = os.path.join(self.test_dir, "features_matrix.parquet")
        
        # --- GENERAR DATOS DUMMY (Con todas las capas) ---
        # 2 Tickers, 10 días
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        data = []
        
        # Generamos datos variados para probar filtros
        for ticker in ["AAPL", "MSFT"]:
            for d in dates:
                data.append({
                    "Date": d,
                    "ticker": ticker,
                    "sector": "Tech",
                    "country": "US",
                    "Close": 150.0,
                    # Raw
                    "rsi_14": 50.0,
                    # Robust
                    "rsi_14_rob": 0.5,
                    # Neutral
                    "rsi_14_neutral": 0.1,
                    # Columnas "basura" que no deberían salir en capas filtradas
                    "ignore_me": 999,
                    # Metadata extra
                    "Volume": 1000.0,
                    "data_quality": 1
                })
        
        self.df_dummy = pl.DataFrame(data).with_columns([
            pl.col("Date").cast(pl.Datetime),
            pl.col("Close").cast(pl.Float64)
        ])
        
        # Guardamos el parquet inicial
        self._write_fake_parquet()

    def _write_fake_parquet(self):
        """Helper para escribir el parquet físico."""
        self.df_dummy.write_parquet(self.fake_parquet_path)

    def tearDown(self):
        """Limpieza post-test."""
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    @patch("master_features.pipeline_features.run_pipeline")
    def test_cold_start_execution(self, mock_pipeline):
        """
        Escenario: El archivo NO existe.
        Flujo esperado: Master detecta ausencia -> Llama Pipeline -> Pipeline crea archivo -> Master lee archivo.
        """
        print("\n🧪 Test Master: Cold Start (Generación Automática)...")
        
        # 1. Borramos el archivo para simular que no existe
        if os.path.exists(self.fake_parquet_path):
            os.remove(self.fake_parquet_path)
            
        # 2. Configuración del Mock:
        # Cuando master llame a run_pipeline, este mock ejecutará _write_fake_parquet
        # simulando que el pipeline real ha trabajado y guardado el archivo.
        mock_pipeline.side_effect = self._write_fake_parquet
        
        # 3. Ejecución
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            df = master_features.get_feature_matrix(layer="raw")
            
            # Validaciones
            mock_pipeline.assert_called_once()  # ¿Se llamó al pipeline?
            self.assertFalse(df.is_empty())     # ¿Se leyeron los datos generados?
            print("   ✅ Flujo completo: Detección -> Generación -> Lectura exitoso.")

    def test_filtering_precision(self):
        """Verifica filtros de Ticker y Fechas con precisión."""
        print("🧪 Test Master: Filtros de Lectura...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # Caso 1: Filtro Ticker + Rango Fecha
            df = master_features.get_feature_matrix(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-02"
            )
            
            # Debe haber 2 filas (dias 1 y 2) de AAPL
            self.assertEqual(df.height, 2)
            self.assertEqual(df["ticker"][0], "AAPL")
            
            # Validación de Tipos (Crucial en Quant)
            self.assertTrue(df["Date"].dtype in [pl.Datetime, pl.Date], "La fecha debe ser Datetime")
            
            print("   ✅ Filtros y Tipos de datos correctos.")

    def test_empty_result_handling(self):
        """Verifica que NO se rompa si los filtros no devuelven nada."""
        print("🧪 Test Master: Manejo de Resultados Vacíos...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # Pedimos un ticker que no existe
            df = master_features.get_feature_matrix(tickers=["NON_EXISTENT"])
            
            self.assertTrue(isinstance(df, pl.DataFrame))
            self.assertTrue(df.is_empty())
            print("   ✅ Devuelve DataFrame vacío correctamente (Graceful degrade).")

    def test_layer_logic_strict(self):
        """Verifica que las capas (Layers) traigan EXACTAMENTE lo que deben."""
        print("🧪 Test Master: Lógica de Capas (Layers)...")
        
        meta_cols = {"Date", "ticker", "sector", "country", "Close", "Volume", "data_quality"}
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            
            # --- CASO NEUTRAL ---
            df_neu = master_features.get_feature_matrix(layer="neutral")
            cols_neu = set(df_neu.columns)
            
            # Debe tener metadata y columnas _neutral
            self.assertTrue(meta_cols.issubset(cols_neu))
            self.assertIn("rsi_14_neutral", cols_neu)
            # NO debe tener _rob ni raw ni ignore_me
            self.assertNotIn("rsi_14", cols_neu)
            self.assertNotIn("rsi_14_rob", cols_neu)
            self.assertNotIn("ignore_me", cols_neu)
            
            # --- CASO RAW ---
            df_raw = master_features.get_feature_matrix(layer="raw")
            cols_raw = set(df_raw.columns)
            
            self.assertIn("rsi_14", cols_raw)
            self.assertNotIn("rsi_14_neutral", cols_raw)
            
            print("   ✅ Aislamiento de capas verificado estrictamente.")

if __name__ == '__main__':
    unittest.main()