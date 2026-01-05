import unittest
import os
import sys
import shutil
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.dirname(current_dir)      # src/engine/features
src_features_dir = os.path.join(features_dir, "src_features")

sys.path.append(src_features_dir)

# Importamos el módulo a testear
import master_features
# Importamos config para parchear las rutas
import config

class TestMasterFeatures(unittest.TestCase):

    def setUp(self):
        """Preparamos un entorno aislado con datos falsos."""
        self.test_dir = os.path.join(current_dir, "temp_master_test")
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.fake_parquet_path = os.path.join(self.test_dir, "features_matrix.parquet")
        
        # --- GENERAR DATOS DUMMY (Con todas las capas) ---
        # 2 Tickers, 10 días
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        data = []
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
                    "vol_20": 0.02,
                    # Robust
                    "rsi_14_rob": 0.5,
                    "vol_20_rob": 1.2,
                    # Neutral
                    "rsi_14_neutral": 0.1,
                    "vol_20_neutral": -0.3,
                    # Columnas extra que no deberían salir en capas filtradas
                    "ignore_me": 999
                })
        
        self.df_dummy = pl.DataFrame(data).with_columns([
            pl.col("Date").cast(pl.Datetime)
        ])
        
        # Guardamos el parquet falso para que el master lo lea
        self.df_dummy.write_parquet(self.fake_parquet_path)

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    @patch("master_features.pipeline_features.run_pipeline")
    def test_auto_update_trigger(self, mock_pipeline):
        """
        Verifica que si el archivo no existe, el Master llama al Pipeline automáticamente.
        """
        print("\n🧪 Test Master: Auto-Update Trigger...")
        
        # 1. Borramos el archivo intencionalmente
        os.remove(self.fake_parquet_path)
        
        # 2. Configurar patch para que el master apunte a nuestra ruta borrada
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # Intentamos leer
            # (Fallará la lectura real porque el mock del pipeline no crea el archivo, 
            # pero lo que queremos ver es si intentó llamarlo)
            try:
                _ = master_features.get_feature_matrix()
            except Exception:
                pass # Es esperado que falle al leer si el mock no genera nada
            
            # VERIFICACIÓN: ¿Llamó al pipeline?
            mock_pipeline.assert_called_once()
            print("   ✅ El Master detectó la ausencia de archivo e invocó al Pipeline.")

    def test_filtering_logic(self):
        """Verifica que los filtros de Fecha y Ticker funcionan."""
        print("🧪 Test Master: Filtros de Lectura...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # Filtramos solo 1 ticker y primeros 5 días
            df = master_features.get_feature_matrix(
                tickers="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-05"
            )
            
            self.assertEqual(df["ticker"].unique().item(), "AAPL")
            self.assertEqual(df.height, 5)
            print("   ✅ Filtros aplicados correctamente.")

    def test_layer_selection(self):
        """Verifica que el parámetro 'layer' devuelve las columnas correctas."""
        print("🧪 Test Master: Selección de Capas (Layers)...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # CASO A: Layer NEUTRAL
            df_neu = master_features.get_feature_matrix(layer="neutral")
            cols_neu = df_neu.columns
            
            self.assertIn("rsi_14_neutral", cols_neu)
            self.assertNotIn("rsi_14", cols_neu, "No debería traer columnas raw en modo neutral")
            self.assertNotIn("rsi_14_rob", cols_neu)
            self.assertIn("Close", cols_neu, "Siempre debe traer precios")
            
            # CASO B: Layer ROBUST
            df_rob = master_features.get_feature_matrix(layer="robust")
            cols_rob = df_rob.columns
            
            self.assertIn("rsi_14_rob", cols_rob)
            self.assertNotIn("rsi_14_neutral", cols_rob)
            
            # CASO C: Layer RAW
            df_raw = master_features.get_feature_matrix(layer="raw")
            cols_raw = df_raw.columns
            
            self.assertIn("rsi_14", cols_raw)
            self.assertNotIn("rsi_14_rob", cols_raw)
            
            print("   ✅ Sistema de capas (Raw/Robust/Neutral) funciona perfecto.")

if __name__ == '__main__':
    unittest.main()