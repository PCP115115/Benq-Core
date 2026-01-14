import unittest
import torch
import numpy as np
import polars as pl
import sys
import os
import time
import shutil
import tempfile
from torch.utils.data import DataLoader, TensorDataset

# --- AJUSTE DE PATH PARA IMPORTAR MÓDULOS DEL PROYECTO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(project_root)

# Importamos la clase a testear
from src.engine.context.auto_encoder_lstm import LSTMHandler
# Importamos la configuración REAL centralizada
from src.engine.config import CONTEXT_PARAMS

class TestLSTMAutoEncoder(unittest.TestCase):
    
    def setUp(self):
        """
        Configuración inicial:
        Carga los parámetros REALES de config.py pero adapta lo necesario
        para que el test sea rápido (mocking parcial).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Cargar configuración de producción
        # Usamos .copy() para no alterar la variable global durante los tests
        self.params = CONTEXT_PARAMS.copy()
        
        # 2. Sobrescribir parámetros solo para el entorno de TEST
        # No queremos entrenar 50 épocas en un unit test, con 2 basta para verificar que corre.
        self.params["LSTM_EPOCHS"] = 2 
        
        # 3. Extraer features definidas en config
        self.features = self.params["INPUT_FEATURES"]
        self.input_dim = len(self.features)
        self.window_size = self.params["LSTM_WINDOW_SIZE"]
        
        # 4. Generar DataFrame Sintético compatible con la config real
        # Creamos 150 filas para asegurar que sea mayor que cualquier ventana típica
        n_rows = 150
        np.random.seed(42)
        
        # Generamos datos aleatorios correspondientes a las features reales (ej: vol_yz, ker_10...)
        data = np.random.randn(n_rows, self.input_dim)
        
        # IMPORTANTE: Usamos los nombres reales de las columnas definidos en config
        self.df = pl.DataFrame(data, schema=self.features)
        
        # Instanciamos el Handler con la config real
        self.handler = LSTMHandler(input_dim=self.input_dim, config_params=self.params)

    def test_01_data_preparation_logic(self):
        """
        Valida que el stride trick funcione con la WINDOW_SIZE real de config.py
        """
        print(f"\n--- Test 01: Preparación de Datos (Window: {self.window_size}) ---")
        
        start_time = time.time()
        tensor_x, _ = self.handler.prepare_data(self.df, self.features)
        exec_time = time.time() - start_time
        
        print(f"Tiempo de ejecución: {exec_time:.6f}s")
        
        # Validación dinámica basada en config
        total_rows = self.df.height
        expected_rows = total_rows - self.window_size + 1
        expected_shape = (expected_rows, self.window_size, self.input_dim)
        
        self.assertEqual(tensor_x.shape, expected_shape, 
                         f"Dimensiones incorrectas usando config real. Esperado: {expected_shape}, Obtenido: {tensor_x.shape}")

    def test_02_model_architecture_shapes(self):
        """
        Valida que el modelo se construya con las dimensiones ocultas y latentes reales.
        """
        print(f"\n--- Test 02: Arquitectura (Latent Dim: {self.params['LSTM_LATENT_DIM']}) ---")
        
        batch_size = 32
        dummy_input = torch.randn(batch_size, self.window_size, self.input_dim).to(self.device)
        
        reconstruction, latent = self.handler.model(dummy_input)
        
        # Verificar contra config real
        expected_latent_shape = (batch_size, self.params["LSTM_LATENT_DIM"])
        self.assertEqual(latent.shape, expected_latent_shape,
                         f"Dimensión latente no coincide con config.py")
        
        self.assertEqual(reconstruction.shape, dummy_input.shape,
                         "La reconstrucción debe tener la misma forma que la entrada.")

    def test_03_training_logic_real_config(self):
        """
        Prueba el loop de entrenamiento usando los hiperparámetros reales (LR, Layers),
        excepto Epochs que redujimos para velocidad.
        """
        print("\n--- Test 03: Entrenamiento con Configuración de Producción ---")
        
        tensor_x, _ = self.handler.prepare_data(self.df, self.features)
        
        # DataLoader dummy
        dataset = TensorDataset(tensor_x)
        train_loader = DataLoader(dataset, batch_size=self.params["LSTM_BATCH_SIZE"], shuffle=False)
        
        initial_weights = self.handler.model.encoder_linear.weight.data.clone()
        
        # Fit con la lógica anti-lookahead
        self.handler.fit(train_loader, val_split=0.2)
        
        final_weights = self.handler.model.encoder_linear.weight.data
        self.assertFalse(torch.equal(initial_weights, final_weights), 
                         "El modelo no está aprendiendo (pesos estáticos).")

    def test_04_encoding_inference(self):
        """
        Verifica inferencia usando features reales.
        """
        print("\n--- Test 04: Inferencia (Encode) ---")
        tensor_x, _ = self.handler.prepare_data(self.df, self.features)
        latent = self.handler.encode(tensor_x)
        
        self.assertEqual(latent.shape[1], self.params["LSTM_LATENT_DIM"])

    def test_05_save_load_system(self):
        """
        Verifica persistencia en ruta temporal.
        """
        print("\n--- Test 05: Save/Load ---")
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "test_lstm_config.pth")
            self.handler.save(path)
            
            # Recargar
            new_handler = LSTMHandler(self.input_dim, self.params)
            new_handler.load(path)
            
            w1 = self.handler.model.encoder_linear.weight.data
            w2 = new_handler.model.encoder_linear.weight.data
            self.assertTrue(torch.equal(w1, w2))

if __name__ == '__main__':
    unittest.main()