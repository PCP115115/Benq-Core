import unittest
import os
import sys
import shutil
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# --- SETUP DE RUTAS ---
# Ajustamos paths para poder importar módulos hermanos y padres
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.dirname(current_dir)      # src/engine/features
engine_dir = os.path.dirname(features_dir)       # src/engine
src_dir = os.path.dirname(engine_dir)            # src

sys.path.append(features_dir)    # Para importar src_features
sys.path.append(src_dir)         # Para importar src_DD
sys.path.append(os.path.join(features_dir, "src_features"))

# Importamos el pipeline bajo prueba
import pipeline_features
# Importamos config para parchearlo
import config

class TestFeaturePipeline(unittest.TestCase):

    def setUp(self):
        """Configuración previa a cada test: Entorno temporal."""
        self.test_dir = os.path.join(current_dir, "temp_pipeline_test")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Ruta simulada para el output
        self.fake_output_path = os.path.join(self.test_dir, "test_features.parquet")
        
        # --- GENERACIÓN DE DATOS SINTÉTICOS ---
        # Creamos un micromercado para probar la neutralización
        # 2 Sectores, 4 Tickers (2 por sector), ~150 días
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(150)]
        
        data = []
        tickers_setup = [
            ("AAPL", "Tech", 150.0),    # US
            ("MSFT", "Tech", 300.0),    # US
            ("SAN.MC", "Banks", 4.0),   # ES (Prueba inferencia país)
            ("BBVA.MC", "Banks", 8.0)   # ES
        ]
        
        rng = np.random.default_rng(42)
        
        for tick, sect, base_price in tickers_setup:
            for d in dates:
                # Caminata aleatoria simple
                noise = rng.normal(0, 1)
                price = base_price + noise
                data.append({
                    "Date": d,
                    "ticker": tick,
                    "sector": sect,
                    "Close": abs(price),
                    "Open": abs(price * 0.99),
                    "High": abs(price * 1.05),
                    "Low": abs(price * 0.95),
                    "Volume": rng.integers(1000, 50000),
                    "data_quality": 1
                })
                
        self.df_mock = pl.DataFrame(data).with_columns([
            pl.col("Date").cast(pl.Datetime),
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Float64)
        ])

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_country_inference(self):
        """Prueba unitaria de la lógica de detección de país."""
        print("\n🧪 Test Pipeline: Inferencia de País...")
        
        df = pl.DataFrame({"ticker": ["AAPL", "TEF.MC", "VOW.DE", "7203.T", "LVMH.PA", "GBPUSD=X"]})
        
        res = df.with_columns(pipeline_features.infer_country("ticker"))
        
        # Validaciones
        rows = res.rows_by_key("ticker")
        
        # CORRECCIÓN 1: rows_by_key elimina la columna clave, así que 'country' pasa a ser el índice 0
        self.assertEqual(rows["AAPL"][0][0], "US")
        self.assertEqual(rows["TEF.MC"][0][0], "ES")
        self.assertEqual(rows["VOW.DE"][0][0], "DE")
        self.assertEqual(rows["7203.T"][0][0], "JP")
        self.assertEqual(rows["LVMH.PA"][0][0], "FR")
        print("   ✅ Lógica de sufijos correcta.")

    @patch('pipeline_features.MarketLoader')
    def test_e2e_pipeline_execution(self, MockLoader):
        """
        Prueba de Integración: Ejecuta todo el pipeline con datos mockeados.
        Verifica: Carga -> Cálculo -> Normalización -> Guardado.
        """
        print("\n🚀 Test Pipeline: Ejecución End-to-End...")
        
        # 1. Configurar el Mock del Loader
        mock_instance = MockLoader.return_value
        # get_all_data devuelve nuestro DF sintético
        mock_instance.get_all_data.return_value = self.df_mock
        
        # 2. Parchear la configuración de rutas y parámetros para el test
        test_params = config.FEATURES_PARAMS.copy()
        test_params["SKEW_WINDOW"] = 10 
        test_params["SMA_SLOW"] = 20
        
        test_norm_params = config.NORMALIZATION_PARAMS.copy()
        test_norm_params["ROLLING_WINDOW"] = 30 
        
        test_paths = {"FEATURES_OUTPUT": self.fake_output_path}

        # CORRECCIÓN 2: 'project_root' está en pipeline_features, NO en config
        with patch.dict(config.FEATURES_PARAMS, test_params), \
             patch.dict(config.NORMALIZATION_PARAMS, test_norm_params), \
             patch.dict(config.PATHS, test_paths), \
             patch('pipeline_features.project_root', self.test_dir): 
            
            # --- EJECUTAR PIPELINE ---
            pipeline_features.run_pipeline()
            
            # --- VERIFICACIONES ---
            
            # A) ¿Se creó el archivo?
            self.assertTrue(os.path.exists(self.fake_output_path), "❌ El archivo Parquet no se generó.")
            
            # Leemos el resultado
            df_result = pl.read_parquet(self.fake_output_path)
            cols = df_result.columns
            
            print(f"   📊 Resultado generado: {df_result.height} filas, {len(cols)} columnas.")
            
            # B) Verificación de Columnas Generadas
            self.assertIn("country", cols, "Falta columna 'country'")
            
            # Indicadores Raw (Ejemplo RSI)
            rsi_col = f"rsi_{test_params['RSI_PERIOD']}"
            self.assertIn(rsi_col, cols, "Falta indicador Raw (RSI)")
            
            # Capa Robusta (_rob)
            rsi_rob = f"{rsi_col}_rob"
            self.assertIn(rsi_rob, cols, "Falta normalización temporal (_rob)")
            
            # Capa Neutral (_neutral)
            rsi_neu = f"{rsi_col}_neutral"
            self.assertIn(rsi_neu, cols, "Falta neutralización sectorial (_neutral)")
            
            # C) Verificación de Lógica de Valores
            sample_es = df_result.filter(pl.col("ticker") == "SAN.MC").select("country").head(1).item()
            self.assertEqual(sample_es, "ES")
            
            self.assertFalse(df_result.is_empty())
            
            print("   ✅ Pipeline completado y estructura de datos validada.")

if __name__ == '__main__':
    unittest.main()