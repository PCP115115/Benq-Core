import unittest
import torch
import numpy as np
import polars as pl
import shutil
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustamos para que pueda importar los módulos del engine desde test_context
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.dirname(os.path.dirname(current_dir)) # src/
sys.path.append(engine_dir)

# Importamos las clases a testear
from engine.context.auto_encoder_lstm import LSTMHandler, LSTMAutoEncoder

class TestLSTMAutoEncoder(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("🧪 TEST SUITE: LSTM AUTOENCODER (CON PURGED SPLIT)")
        print("="*60)
        
        # Directorio temporal para outputs
        cls.test_dir = os.path.join(current_dir, "temp_lstm_test_output")
        os.makedirs(cls.test_dir, exist_ok=True)

    def setUp(self):
        # 1. Configuración Dummy del Modelo
        self.params = {
            "LSTM_HIDDEN_DIM": 16,
            "LSTM_LATENT_DIM": 4,
            "LSTM_LAYERS": 1,
            "LSTM_WINDOW_SIZE": 10, # Ventana pequeña para facilitar conteo
            "LSTM_LR": 0.01,
            "LSTM_EPOCHS": 2,       # Pocas épocas para velocidad
            "LSTM_BATCH_SIZE": 16
        }
        
        self.input_dim = 2 # Dos features
        self.handler = LSTMHandler(self.input_dim, self.params)
        
        # 2. Datos Sintéticos (Polars)
        # Generamos suficientes datos para sobrevivir al Purged Split
        # Necesitamos: Window + Gap + Train + Val
        n_rows = 200 
        self.df_mock = pl.DataFrame({
            "feature_1": np.random.randn(n_rows),
            "feature_2": np.random.randn(n_rows)
        })
        self.features = ["feature_1", "feature_2"]

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            print("\n🧹 Limpieza de archivos temporales completada.")

    def test_01_architecture_integrity(self):
        """Verifica que la red neuronal se construye con las dimensiones correctas."""
        print("\n🔹 Test 01: Integridad de Arquitectura")
        model = self.handler.model
        
        # Verificar dimensiones
        self.assertEqual(model.encoder_lstm.input_size, self.input_dim)
        self.assertEqual(model.encoder_lstm.hidden_size, self.params["LSTM_HIDDEN_DIM"])
        self.assertEqual(model.encoder_linear.out_features, self.params["LSTM_LATENT_DIM"])
        
        print("   ✅ Dimensiones de capas verificadas.")

    def test_02_data_preparation_shape(self):
        """Verifica que prepare_data genera ventanas deslizantes correctas (Tensores 3D)."""
        print("\n🔹 Test 02: Preparación de Datos (Ventanas Deslizantes)")
        
        tensor_x, _ = self.handler.prepare_data(self.df_mock, self.features)
        
        # Cálculo esperado: N_samples = Total_Rows - Window + 1
        expected_samples = self.df_mock.height - self.params["LSTM_WINDOW_SIZE"] + 1
        expected_shape = (expected_samples, self.params["LSTM_WINDOW_SIZE"], self.input_dim)
        
        self.assertEqual(tensor_x.shape, expected_shape)
        print(f"   ✅ Shape Correcto: {tensor_x.shape}")

    def test_03_purged_split_logic_enforcement(self):
        """
        CRÍTICO: Verifica que el Purged Split se aplica y protege contra Look-Ahead Bias.
        Intentamos entrenar con datos insuficientes para ver si salta la protección.
        """
        print("\n🔹 Test 03: Lógica de 'Purged Split' (Protección Anti-Leakage)")
        
        # Caso A: Datos INSUFICIENTES para el gap
        # Si tenemos 20 datos y ventana 10:
        # Muestras totales = 20 - 10 + 1 = 11 ventanas.
        # Val split 20% -> Val empieza en idx 8 (aprox).
        # Purge Gap = 10.
        # Train End = 8 - 10 = -2 (IMPOSIBLE) -> Debe fallar.
        
        df_tiny = self.df_mock.head(20) 
        tensor_tiny, _ = self.handler.prepare_data(df_tiny, self.features)
        
        # Creamos un loader dummy
        loader_tiny = DataLoader(TensorDataset(tensor_tiny), batch_size=2)
        
        print("   > Probando con dataset insuficiente para forzar error de seguridad...")
        with self.assertRaises(ValueError) as context:
            self.handler.fit(loader_tiny, val_split=0.2)
        
        self.assertIn("Dataset insuficiente para Purged Split", str(context.exception))
        print("   ✅ El sistema BLOQUEÓ correctamente un entrenamiento con solapamiento temporal (Look-Ahead prevented).")

    def test_04_training_execution(self):
        """Verifica que el entrenamiento (fit) corre de principio a fin con datos válidos."""
        print("\n🔹 Test 04: Ejecución de Entrenamiento (Smoke Test)")
        
        tensor_x, _ = self.handler.prepare_data(self.df_mock, self.features)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=self.params["LSTM_BATCH_SIZE"])
        
        try:
            self.handler.fit(loader, val_split=0.2)
            print("   ✅ Entrenamiento completado sin errores.")
        except Exception as e:
            self.fail(f"El entrenamiento falló con una excepción no esperada: {e}")

    def test_05_encoding_latent_space(self):
        """Verifica que el encoder produce vectores latentes de la dimensión correcta."""
        print("\n🔹 Test 05: Inferencia (Encoding)")
        
        tensor_x, _ = self.handler.prepare_data(self.df_mock.head(50), self.features)
        
        latents = self.handler.encode(tensor_x)
        
        # Shape esperado: (N_samples, LATENT_DIM)
        expected_dim = self.params["LSTM_LATENT_DIM"]
        self.assertEqual(latents.shape[1], expected_dim)
        self.assertEqual(latents.shape[0], tensor_x.shape[0])
        
        print(f"   ✅ Vectores latentes generados correctamente: {latents.shape}")

    def test_06_save_and_load(self):
        """Verifica la persistencia del modelo."""
        print("\n🔹 Test 06: Guardado y Carga de Modelos")
        
        path = os.path.join(self.test_dir, "lstm_test_model.pth")
        
        # Guardar
        self.handler.save(path)
        self.assertTrue(os.path.exists(path), "El archivo del modelo no se creó.")
        
        # Cargar
        try:
            self.handler.load(path)
            print("   ✅ Modelo guardado y cargado correctamente.")
        except Exception as e:
            self.fail(f"Fallo al cargar el modelo: {e}")

if __name__ == '__main__':
    unittest.main()