import unittest
import time
import sys
import os
import polars as pl
from pathlib import Path
from functools import wraps

# ==========================================
# 1. CONFIGURACIÓN DE ENTORNO (PATH HELL FIX)
# ==========================================
# Nos aseguramos de que Python vea la raíz del proyecto 'Benq-Core'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

# ==========================================
# 2. IMPORTACIONES DEL MOTOR
# ==========================================
try:
    import src.engine.config as config
    from src.engine.config import META_MODEL_PARAMS
    from src.engine.src_features.master_features import get_feature_matrix
    from src.engine.context.master_context import get_market_regime
    from src.engine.mini_models.src_mini_models.master_mini_models import run_mini_models_pipeline
    from src.engine.meta_model.src_meta_model.download_meta import get_data_meta_model
except ImportError as e:
    print(f"❌ Error Crítico de Importación: {e}")
    print(f"Ruta intentada: {project_root}")
    sys.exit(1)

# ==========================================
# 3. UTILIDADES DE TEST (Decoradores)
# ==========================================
def timer(func):
    """Decorador para medir tiempo de ejecución de tests individuales."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"   ⏱️ Tiempo de ejecución [{func.__name__}]: {execution_time:.4f} segundos")
        return result
    return wrapper

# ==========================================
# 4. CLASE DE TEST RIGUROSO
# ==========================================
class TestMetaModelPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial compartida."""
        print("\n" + "="*60)
        print("🚀 INICIANDO TEST SUITE: META-MODEL PIPELINE")
        print("="*60)
        
        # Parámetros de prueba
        cls.ticker = "AAPL" # Usamos un ticker líquido y fiable
        cls.start_date = "2020-01-01"
        cls.end_date = "2023-12-31"
        cls.layer = "neutral"
        cls.features = META_MODEL_PARAMS["feature_list"]
        cls.norm_window = META_MODEL_PARAMS["normalization_window"]
        
        # Ruta esperada del output de mini-modelos
        cls.mini_model_path = os.path.join(
            project_root, 
            "src", "data", "processed", "meta_model_inputs", 
            f"meta_input_{cls.ticker}.parquet"
        )

    def setUp(self):
        print(f"\n🔹 Ejecutando: {self._testMethodName}...")

    @timer
    def test_01_features_module_performance(self):
        """Evalúa rendimiento y consistencia del Módulo de Features."""
        df = get_feature_matrix(
            tickers=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            layer=self.layer,
            features=self.features,
            normalization_window=self.norm_window
        )
        
        # Aserciones Rigurosas
        self.assertIsInstance(df, pl.DataFrame, "El output debe ser un DataFrame de Polars")
        self.assertFalse(df.is_empty(), "El DataFrame de Features no puede estar vacío")
        
        # Verificar columnas clave
        expected_cols = ["Date", "ticker"] + self.features
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Falta la columna crítica: {col}")
            
        # Verificar tipos
        self.assertEqual(df["Date"].dtype, pl.Datetime, "La columna Date debe ser Datetime")

    @timer
    def test_02_context_module_performance(self):
        """Evalúa rendimiento del Módulo de Contexto (LSTM + GMM)."""
        # Nota: Pasamos lista [self.ticker] porque el contexto soporta batch
        df = get_market_regime(tickers=[self.ticker])
        
        self.assertFalse(df.is_empty(), "El DataFrame de Contexto devolvió vacío (posible fallo en carga de modelos)")
        
        # Verificar Regímenes
        self.assertIn("market_regime", df.columns)
        self.assertIn("regime_probability", df.columns)
        
        # Verificar que los regímenes son válidos (0 a N_COMPONENTS-1)
        unique_regimes = df["market_regime"].unique().to_list()
        self.assertTrue(all(isinstance(x, (int, float)) for x in unique_regimes))

    @timer
    def test_03_mini_models_execution(self):
        """Evalúa la ejecución del Pipeline de Expertos (Mini-Modelos)."""
        # Ejecutamos el pipeline
        run_mini_models_pipeline(tickers=[self.ticker], n_jobs=1)
        
        # Validación de Persistencia (¿Se guardó el archivo?)
        self.assertTrue(os.path.exists(self.mini_model_path), 
                        f"No se generó el archivo de salida en: {self.mini_model_path}")
        
        # Validación de Lectura
        df = pl.read_parquet(self.mini_model_path)
        self.assertFalse(df.is_empty())
        
        # Verificar existencia de columnas de Probabilidad (Output de los modelos)
        prob_cols = [c for c in df.columns if c.startswith("P_")]
        self.assertTrue(len(prob_cols) > 0, "No se encontraron columnas de probabilidad (P_...) en el output")

    @timer
    def test_04_full_integration_pipeline(self):
        """TEST FINAL: Integración completa (Download Meta)."""
        df_final = get_data_meta_model(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            layer=self.layer,
            feature_list=self.features,
            normalization_window=self.norm_window
        )
        
        # 1. Integridad Estructural
        self.assertFalse(df_final.is_empty(), "El Dataset Final del Meta-Modelo está vacío")
        
        # 2. Validación de Joins (Inner Join no debe matar todos los datos)
        # Debería tener Features + Contexto + MiniModelos
        expected_min_cols = len(self.features) + 2 + 1 # Features + Date/Ticker + Context
        self.assertGreaterEqual(len(df_final.columns), expected_min_cols)
        
        # 3. Validación de Fechas
        # Las fechas deben ser únicas por ticker y estar ordenadas
        dates = df_final["Date"]
        self.assertTrue(dates.is_sorted(), "Las fechas no están ordenadas cronológicamente")
        n_rows = df_final.height
        n_unique_dates = df_final["Date"].n_unique()
        self.assertEqual(n_rows, n_unique_dates, "Hay duplicados de fechas para el mismo ticker")

        print(f"   ✅ Dataset Final: {n_rows} filas válidas generadas.")

if __name__ == "__main__":
    unittest.main()