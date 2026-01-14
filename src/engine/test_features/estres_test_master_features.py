import unittest
import os
import sys
import shutil
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# --- SETUP DE RUTAS ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(current_test_dir)
src_features_dir = os.path.join(engine_dir, "src_features")

if src_features_dir not in sys.path:
    sys.path.append(src_features_dir)

# Importamos el módulo a testear
import master_features

class TestMasterFeaturesStress(unittest.TestCase):

    def setUp(self):
        """Preparación del entorno antes de cada test."""
        self.test_dir = os.path.join(current_test_dir, "temp_master_stress")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Ruta del archivo dummy
        self.dummy_parquet_path = os.path.join(self.test_dir, "features_matrix.parquet")
        
        # Generamos datos sintéticos representativos
        self.df_dummy = self._generate_dummy_data()
        self.df_dummy.write_parquet(self.dummy_parquet_path)

        # Patching crítico: Forzamos a master_features a usar nuestro archivo temporal
        self.patcher = patch('master_features.DATA_PATH', self.dummy_parquet_path)
        self.patcher.start()

    def tearDown(self):
        """Limpieza después de cada test."""
        self.patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _generate_dummy_data(self):
        """Genera un DataFrame completo con estructura de producción."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        tickers = ["AAPL", "GOOGL", "TSLA"]
        
        data = []
        for t in tickers:
            # Datos Raw
            close = np.random.normal(100, 10, 100)
            rsi = np.random.uniform(20, 80, 100)
            vol = np.random.normal(0.02, 0.01, 100)
            
            df = pl.DataFrame({
                "Date": dates,
                "ticker": [t] * 100,
                "sector": ["Tech"] * 100,
                "country": ["US"] * 100,
                "data_quality": [1] * 100,
                "Close": close,
                "Open": close, "High": close, "Low": close, "Volume": close * 1000,
                
                # Features Raw
                "rsi": rsi,
                "volatility": vol,
                
                # Features Robust (Mocked values)
                "rsi_rob": (rsi - 50) / 10,
                "volatility_rob": (vol - 0.02) / 0.01,
                
                # Features Neutral (Mocked values)
                "rsi_neutral": (rsi - 50) / 15,
                "volatility_neutral": (vol - 0.02) / 0.015
            })
            data.append(df)
        
        return pl.concat(data)

    # -------------------------------------------------------------------------
    # TEST 1: COLD START & AUTO-HEALING
    # -------------------------------------------------------------------------
    @patch('master_features.pipeline_features.run_pipeline')
    def test_cold_start_execution(self, mock_pipeline):
        """Verifica que si no existe el archivo, se llama al pipeline."""
        # Borramos el archivo creado en setUp
        os.remove(self.dummy_parquet_path)
        
        # Ejecutamos update explicitamente para simular la regeneración
        # (Mockeamos run_pipeline para que no tarde años ejecutando el real)
        master_features.update_features()
        
        mock_pipeline.assert_called_once()
        print("\n✅ [Cold Start] El sistema intentó regenerar los features ante la ausencia de datos.")

    # -------------------------------------------------------------------------
    # TEST 2: FILTROS DE TIEMPO Y TICKER
    # -------------------------------------------------------------------------
    def test_basic_filtering(self):
        """Prueba exhaustiva de filtros de filas."""
        # Caso A: Un solo ticker
        df = master_features.get_feature_matrix(tickers="AAPL")
        self.assertEqual(df["ticker"].n_unique(), 1)
        self.assertEqual(df["ticker"][0], "AAPL")
        
        # Caso B: Rango de fechas
        start = "2023-01-10"
        end = "2023-01-20"
        df = master_features.get_feature_matrix(start_date=start, end_date=end)
        
        self.assertTrue(df["Date"].min() >= datetime(2023, 1, 10))
        self.assertTrue(df["Date"].max() <= datetime(2023, 1, 20))
        
        # Caso C: Ticker inexistente (Debe devolver DataFrame vacío, no error)
        df_empty = master_features.get_feature_matrix(tickers="INVALID_TICKER")
        self.assertTrue(df_empty.is_empty())
        
        print("✅ [Filtros] Tickers y Fechas filtrados correctamente.")

    # -------------------------------------------------------------------------
    # TEST 3: SELECCIÓN DE CAPAS (LAYERS)
    # -------------------------------------------------------------------------
    def test_layer_logic(self):
        """Verifica que se devuelvan las columnas correctas según la capa."""
        # 1. Layer RAW
        df_raw = master_features.get_feature_matrix(layer="raw")
        cols_raw = df_raw.columns
        self.assertIn("rsi", cols_raw)
        self.assertNotIn("rsi_rob", cols_raw)
        self.assertNotIn("rsi_neutral", cols_raw)
        
        # 2. Layer ROBUST
        df_rob = master_features.get_feature_matrix(layer="robust")
        cols_rob = df_rob.columns
        self.assertIn("rsi_rob", cols_rob)
        self.assertNotIn("rsi", cols_rob) # RSI raw no debería estar en robust layer (salvo config específica)
        
        # 3. Layer NEUTRAL
        df_neu = master_features.get_feature_matrix(layer="neutral")
        cols_neu = df_neu.columns
        self.assertIn("rsi_neutral", cols_neu)
        
        print("✅ [Layers] La separación de capas Raw/Robust/Neutral es correcta.")

    # -------------------------------------------------------------------------
    # TEST 4: FILTRO POR KEYWORDS (FEATURES)
    # -------------------------------------------------------------------------
    def test_feature_keyword_filtering(self):
        """Verifica la búsqueda de columnas por string parcial."""
        # Pedimos solo 'volatility' en capa 'neutral'
        df = master_features.get_feature_matrix(layer="neutral", features="volatility")
        
        self.assertIn("volatility_neutral", df.columns)
        self.assertNotIn("rsi_neutral", df.columns) # RSI no debería estar
        self.assertIn("ticker", df.columns) # Las columnas obligatorias siempre deben estar
        
        # Caso Negativo: Keyword que no existe
        with self.assertRaises(ValueError):
            master_features.get_feature_matrix(features="non_existent_feature")
            
        print("✅ [Keywords] El filtrado de columnas por nombre funciona.")

    # -------------------------------------------------------------------------
    # TEST 5: NORMALIZACIÓN DINÁMICA (EL MÁS CRÍTICO)
    # -------------------------------------------------------------------------
    def test_dynamic_normalization(self):
        """Prueba matemática de la normalización 'al vuelo'."""
        window = 10
        
        # Pedimos normalización dinámica de RSI
        df_dyn = master_features.get_feature_matrix(
            tickers="AAPL", 
            features="rsi", 
            normalization_window=window
        )
        
        # Extraemos la serie calculada por el master
        rsi_dynamic = df_dyn["rsi"].to_numpy()
        
        # --- VERIFICACIÓN MATEMÁTICA MANUAL ---
        # Reconstruimos la lógica para validar
        # Tomamos los primeros 20 valores RAW de mi dummy data para AAPL
        df_raw_check = self.df_dummy.filter(pl.col("ticker") == "AAPL").select("rsi").head(20)
        rsi_raw_vals = df_raw_check["rsi"].to_numpy()
        
        # Calculamos manual el último valor (índice 15 por ejemplo)
        idx = 15
        slice_vals = rsi_raw_vals[idx - window + 1 : idx + 1] # Ventana de 10
        
        # NumPy usa linear por defecto, ahora Master Features TAMBIÉN. Deben coincidir.
        median = np.median(slice_vals)
        q75, q25 = np.percentile(slice_vals, [75 ,25]) # Linear interpolation implicit
        iqr = q75 - q25
        expected_z = (rsi_raw_vals[idx] - median) / iqr
        
        # Obtenemos el valor que calculó el master
        master_val = df_dyn["rsi"][idx]
        
        # Tolerancia pequeña por diferencias de punto flotante
        np.testing.assert_almost_equal(master_val, expected_z, decimal=5)
        
        total_nulls = df_dyn.null_count().sum_horizontal().item()
        self.assertEqual(total_nulls, 0)

        print(f"✅ [Matemática] La normalización dinámica (Rolling Z-Score {window}d) es exacta.")
if __name__ == '__main__':
    unittest.main()