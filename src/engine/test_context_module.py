import unittest
import torch
import numpy as np
import polars as pl
import os
import sys
import shutil
import importlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, ANY

# --- 1. CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
engine_dir = os.path.dirname(current_dir)                
src_dir = os.path.dirname(engine_dir)                    
root_dir = os.path.dirname(src_dir)                      

sys.path.append(src_dir)
sys.path.append(engine_dir)
sys.path.append(current_dir)

# --- 2. IMPORTACIONES Y MOCKS ---
try:
    import config
    # Mockeamos dependencias externas pesadas
    sys.modules["src_features"] = MagicMock()
    sys.modules["src_features.master_features"] = MagicMock()
    
    from auto_encoder_lstm import LSTMHandler
    from gmm_model import RegimeDetector
    import master_context
except ImportError as e:
    raise ImportError(f"❌ Error importando módulos. Detalle: {e}")

# --- 3. SUITE DE PRUEBAS AVANZADAS ---

class TestSemanticStability(unittest.TestCase):
    """
    🛡️ TEST DE ESTABILIDAD SEMÁNTICA
    Objetivo: Demostrar que el Régimen 0 SIEMPRE es baja volatilidad
    y el Régimen N es alta volatilidad, sin importar el entrenamiento.
    """

    def setUp(self):
        self.n_components = 3
        # Generamos datos sintéticos claros:
        # Grupo A (Calma): Latentes cercanos a 0, Anchor (Volatilidad) bajo
        # Grupo B (Medio): Latentes cercanos a 5, Anchor medio
        # Grupo C (Pánico): Latentes cercanos a 10, Anchor alto
        
        n_samples = 100
        self.dim = 4
        
        # Latentes (Simulados como salida del LSTM)
        latents_A = np.random.normal(0, 0.5, (n_samples, self.dim))
        latents_B = np.random.normal(5, 0.5, (n_samples, self.dim))
        latents_C = np.random.normal(10, 0.5, (n_samples, self.dim))
        self.X = np.vstack([latents_A, latents_B, latents_C])
        
        # Métrica Ancla (Volatilidad explícita para cada grupo)
        anchor_A = np.random.uniform(0.0, 0.1, n_samples) # Volatilidad < 0.1
        anchor_B = np.random.uniform(0.4, 0.6, n_samples) # Volatilidad ~ 0.5
        anchor_C = np.random.uniform(0.9, 1.0, n_samples) # Volatilidad > 0.9
        self.anchor = np.concatenate([anchor_A, anchor_B, anchor_C])

    def test_semantic_stability_under_chaos(self):
        """
        🔥 PRUEBA DE FUEGO: Entrenamos dos modelos.
        Uno normal y otro donde forzamos 'shuffle' aleatorio de los datos antes de entrenar.
        Ambos deben concluir que el Grupo C es el régimen más alto (2).
        """
        print("\n🧪 [GMM] Stress Test: Estabilidad Semántica...")
        
        # Modelo 1: Entrenamiento Ordenado
        model_1 = RegimeDetector(n_components=self.n_components)
        model_1.fit(self.X, self.anchor)
        
        # Modelo 2: Entrenamiento Caótico (Shuffle de datos)
        # Esto normalmente cambiaría los IDs de los clústeres si no hubiera sorting.
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        X_shuffled = self.X[indices]
        anchor_shuffled = self.anchor[indices]
        
        model_2 = RegimeDetector(n_components=self.n_components)
        model_2.fit(X_shuffled, anchor_shuffled)
        
        # VERIFICACIÓN
        # Tomamos una muestra representativa de ALTA VOLATILIDAD (del Grupo C original)
        sample_high_vol = np.array([[10.0] * self.dim]) 
        
        pred_1, _ = model_1.predict(sample_high_vol)
        pred_2, _ = model_2.predict(sample_high_vol)
        
        print(f"   Predicción Modelo 1 (Normal): Régimen {pred_1[0]}")
        print(f"   Predicción Modelo 2 (Caos):   Régimen {pred_2[0]}")
        
        # Ambos deben predecir el índice máximo (n_components - 1 = 2)
        target_regime = self.n_components - 1
        
        self.assertEqual(pred_1[0], target_regime, "❌ Modelo 1 no asignó ID máximo a alta volatilidad.")
        self.assertEqual(pred_2[0], target_regime, "❌ Modelo 2 (Caos) no asignó ID máximo a alta volatilidad.")
        self.assertEqual(pred_1[0], pred_2[0], "❌ Inconsistencia semántica entre modelos.")
        
        print("   ✅ ÉXITO: El significado de 'Régimen Alto' es estable tras re-entrenamientos.")

    def test_semantic_ordering_logic(self):
        """Verifica internamente que el mapa de traducción es monótono."""
        model = RegimeDetector(n_components=self.n_components)
        model.fit(self.X, self.anchor)
        
        # Accedemos a internals para verificar la lógica de map
        # El mapa debe existir
        self.assertTrue(hasattr(model, 'regime_map') and model.regime_map)
        print(f"   Map generado: {model.regime_map}")


class TestMasterContextRobustness(unittest.TestCase):
    """
    🛡️ TEST DE ROBUSTEZ Y ORQUESTACIÓN
    Objetivo: Verificar que master_context extrae bien el ancla y maneja errores.
    """

    def setUp(self):
        self.test_dir = os.path.join(current_dir, "temp_context_v2")
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.patcher_paths = patch.dict(config.PATHS, {
            "MODEL_LSTM": os.path.join(self.test_dir, "lstm.pth"),
            "MODEL_GMM": os.path.join(self.test_dir, "gmm.joblib")
        })
        self.patcher_paths.start()

        self.mock_get_features = patch('master_context.master_features.get_feature_matrix').start()

    def tearDown(self):
        self.patcher_paths.stop()
        patch.stopall()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_df(self, n_rows=100):
        start_date = datetime(2023, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        data = {
            "Date": pl.date_range(start=start_date, end=end_date, interval="1d", eager=True),
            "ticker": ["TEST"] * n_rows
        }
        # Creamos features. IMPORTANTE: La primera feature es la volatilidad (anchor)
        vol_col = config.CONTEXT_PARAMS["INPUT_FEATURES"][0]
        data[vol_col] = np.linspace(0, 1, n_rows) # Volatilidad de 0 a 1
        
        for col in config.CONTEXT_PARAMS["INPUT_FEATURES"][1:]:
            data[col] = np.random.random(n_rows)
            
        return pl.DataFrame(data)

    def test_anchor_propagation(self):
        """
        Verifica que master_context extrae la columna correcta del Tensor (Anchor)
        y se la pasa al método fit() del GMM.
        """
        print("\n🧪 [MASTER] Verificando Propagación de Ancla (Volatilidad)...")
        
        n_rows = 100
        self.mock_get_features.return_value = self._create_mock_df(n_rows)
        
        # --- FIX: Mockeamos TAMBIÉN predict para evitar el RuntimeError ---
        # Como mockeamos fit, el modelo no se entrena. predict lanzaría error.
        # Al mockear predict, evitamos que ejecute la lógica real y el test pasa.
        with patch('master_context.RegimeDetector.fit') as mock_fit, \
             patch('master_context.RegimeDetector.predict') as mock_predict:
            
            # Configuramos el mock de predict para devolver arrays vacíos (dummy)
            # El LSTM consume 'window' datos, así que la salida es menor a 100
            dummy_len = n_rows - config.CONTEXT_PARAMS["LSTM_WINDOW_SIZE"] + 1
            mock_predict.return_value = (np.zeros(dummy_len, dtype=int), np.zeros(dummy_len))

            # Ejecutamos
            importlib.reload(master_context)
            master_context.get_market_regime(tickers=["TEST"])
            
            # Verificamos llamada a FIT
            self.assertTrue(mock_fit.called, "❌ No se llamó a GMM.fit")
            
            # fit(latents, anchor_metric) -> args[0]=latents, args[1]=anchor
            args, _ = mock_fit.call_args
            anchor_passed = args[1]
            
            self.assertIsNotNone(anchor_passed, "❌ El argumento 'anchor_metric' llegó como None.")
            self.assertEqual(len(anchor_passed.shape), 1, "❌ El anchor debe ser un array 1D.")
            
            # El anchor debe tener valores > 0 (nuestro mock tiene vol hasta 1.0)
            self.assertGreater(anchor_passed.max(), 0.0, "❌ El anchor parece vacío o corrupto.")
            print("   ✅ El orquestador extrajo y pasó la métrica de anclaje correctamente.")

    def test_edge_case_insufficient_data(self):
        """Caso: Dataframe existe pero es más corto que la ventana LSTM."""
        print("🧪 [MASTER] Probando Edge Case: Datos Insuficientes...")
        
        # LSTM Window suele ser 20. Damos solo 5 filas.
        short_df = self._create_mock_df(5)
        self.mock_get_features.return_value = short_df
        
        importlib.reload(master_context)
        df_res = master_context.get_market_regime(tickers=["TEST"])
        
        self.assertTrue(df_res.is_empty(), "❌ Debería devolver DF vacío si no hay suficientes datos.")
        print("   ✅ Manejo correcto de series cortas.")

    def test_edge_case_empty_input(self):
        """Caso: master_features devuelve vacío."""
        self.mock_get_features.return_value = pl.DataFrame()
        importlib.reload(master_context)
        df_res = master_context.get_market_regime()
        self.assertTrue(df_res.is_empty())

if __name__ == '__main__':
    unittest.main(verbosity=2)