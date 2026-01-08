import unittest
import os
import sys
import shutil
import polars as pl
from datetime import datetime, timedelta
from unittest.mock import patch

# --- SETUP DE RUTAS ROBUSTO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(engine_dir)

sys.path.append(src_dir)
sys.path.append(engine_dir)
sys.path.append(os.path.join(engine_dir, "src_features"))

import master_features

class TestMasterFeaturesPro(unittest.TestCase):

    def setUp(self):
        """Preparamos un entorno aislado con datos falsos pero estructuralmente ricos."""
        self.test_dir = os.path.join(current_dir, "temp_master_test")
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.fake_parquet_path = os.path.join(self.test_dir, "features_matrix.parquet")
        
        # --- GENERAR DATOS DUMMY (Incluyendo Yang-Zhang y Parkinson Neutral) ---
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        data = []
        
        for ticker in ["AAPL", "MSFT"]:
            for d in dates:
                data.append({
                    "Date": d,
                    "ticker": ticker,
                    "sector": "Tech",
                    "Close": 150.0,
                    
                    # --- Raw Indicators ---
                    "rsi_14": 50.0,
                    "vol_parkinson_20d": 0.02,
                    "vol_yz_20d": 0.025,  
                    
                    # --- Robust Indicators ---
                    "rsi_14_rob": 0.5,
                    "vol_yz_20d_rob": 1.5,
                    
                    # --- Neutral Indicators ---
                    "rsi_14_neutral": 0.1,
                    "vol_yz_20d_neutral": 0.4, 
                    # CORRECCIÓN: Añadimos Parkinson Neutral para que el test de intersección funcione
                    "vol_parkinson_20d_neutral": 0.3, 
                    
                    "ignore_me_temp": 999
                })
        
        self.df_dummy = pl.DataFrame(data).with_columns([
            pl.col("Date").cast(pl.Datetime),
            pl.col("Close").cast(pl.Float64)
        ])
        
        self._write_fake_parquet()

    def _write_fake_parquet(self):
        self.df_dummy.write_parquet(self.fake_parquet_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    @patch("master_features.pipeline_features.run_pipeline")
    def test_cold_start_execution(self, mock_pipeline):
        print("\n🧪 Test Master: Cold Start (Generación Automática)...")
        if os.path.exists(self.fake_parquet_path):
            os.remove(self.fake_parquet_path)
        mock_pipeline.side_effect = self._write_fake_parquet
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            df = master_features.get_feature_matrix(layer="raw")
            mock_pipeline.assert_called_once()
            self.assertFalse(df.is_empty())
            print("   ✅ Cold Start gestionado correctamente.")

    def test_yang_zhang_filtering(self):
        """
        [NUEVO] Verifica que podemos filtrar específicamente la volatilidad YZ.
        """
        print("🧪 Test Master: Filtrado específico de Yang-Zhang...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            # Caso: Quiero solo YZ neutralizado
            df = master_features.get_feature_matrix(
                layer="neutral",
                features=["yz"] # Keyword única
            )
            cols = df.columns
            
            self.assertIn("vol_yz_20d_neutral", cols)
            self.assertNotIn("vol_parkinson_20d_neutral", cols) # Parkinson no tiene 'yz'
            print("   ✅ Filtrado por keyword 'yz' exitoso.")

    def test_feature_selection_intersection(self):
        """Prueba compleja: Layer='neutral' + Features=['vol']."""
        print("🧪 Test Master: Intersección Layer (Neutral) + Feature (Vol)...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            df = master_features.get_feature_matrix(
                layer="neutral",
                features=["vol"]
            )
            cols = df.columns
            
            # Debe traer AMBAS volatilidades porque ambas tienen "vol"
            self.assertIn("vol_parkinson_20d_neutral", cols)
            self.assertIn("vol_yz_20d_neutral", cols) 
            
            self.assertNotIn("rsi_14_neutral", cols)
            print("   ✅ Intersección lógica correcta (YZ + Parkinson detectados).")

    def test_filtering_precision(self):
        """Valida filtros básicos (Ticker/Fecha)."""
        print("🧪 Test Master: Filtros Push-down (Ticker/Fecha)...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            df = master_features.get_feature_matrix(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-01" # Solo 1 día
            )
            
            self.assertEqual(df.height, 1)
            self.assertEqual(df["ticker"][0], "AAPL")
            print("   ✅ Filtros básicos OK.")

    def test_feature_safety_exception(self):
        """Prueba la regla de seguridad: Raise ValueError si pedimos algo que no existe."""
        print("🧪 Test Master: Manejo de Errores (Safety Check)...")
        
        with patch("master_features.DATA_PATH", self.fake_parquet_path):
            with self.assertRaises(ValueError):
                master_features.get_feature_matrix(
                    layer="all",
                    features=["super_indicador"]
                )
            print("   ✅ Se lanzó ValueError correctamente.")

if __name__ == '__main__':
    unittest.main()