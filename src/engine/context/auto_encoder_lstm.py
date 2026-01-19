import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
import os

class LSTMAutoEncoder(nn.Module):
    """
    Arquitectura Encoder-Decoder LSTM para comprimir series temporales.
    Input Shape: (Batch, Sequence_Length, Features)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMAutoEncoder, self).__init__()
        self.seq_len = None 
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # --- ENCODE ---
        _, (h_n, _) = self.encoder_lstm(x) 
        last_hidden = h_n[-1] 
        latent = self.encoder_linear(last_hidden) 
        
        # --- DECODE ---
        decoder_input = self.decoder_linear(latent) 
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1) 
        
        output_lstm, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_layer(output_lstm) 
        
        return reconstruction, latent

class LSTMHandler:
    def __init__(self, input_dim, config_params):
        self.params = config_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LSTMAutoEncoder(
            input_dim=input_dim,
            hidden_dim=self.params["LSTM_HIDDEN_DIM"],
            latent_dim=self.params["LSTM_LATENT_DIM"],
            num_layers=self.params["LSTM_LAYERS"]
        ).to(self.device)
        
    def prepare_data(self, df: pl.DataFrame, features: list) -> tuple[torch.Tensor, np.ndarray]:
        """Convierte Polars DF en Tensores 3D deslizantes usando Numpy Stride Tricks."""
        data_arr = df.select(features).to_numpy()
        
        window = self.params["LSTM_WINDOW_SIZE"]
        if len(data_arr) <= window:
             raise ValueError("Datos insuficientes para la ventana LSTM.")

        shape = (data_arr.shape[0] - window + 1, window, data_arr.shape[1])
        strides = (data_arr.strides[0], data_arr.strides[0], data_arr.strides[1])
        
        X = np.lib.stride_tricks.as_strided(data_arr, shape=shape, strides=strides)
        
        # Nota: Devolvemos el tensor completo aquí. El split se hace en fit/entrenamiento
        # para mantener la firma de la función original intacta.
        return torch.FloatTensor(X).to(self.device), data_arr

    def fit(self, train_loader, val_split=0.2):
        """
        Entrena el modelo.
        MODIFICADO: Aplica PURGED SPLIT para evitar Look-Ahead Bias estricto.
        
        Acepta 'train_loader' que puede ser un DataLoader con todos los datos.
        Dentro se realiza la separación cronológica con 'embargo' (purge) de datos.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["LSTM_LR"])
        criterion = nn.MSELoss()
        
        # Lógica para extraer los tensores del DataLoader original sin romper la firma
        # Asumimos que train_loader es un iterable (DataLoader) que contiene TODO el dataset
        all_data_tensors = []
        for batch in train_loader:
            all_data_tensors.append(batch[0])
        
        # Concatenamos para tener el tensor completo (preservando orden temporal)
        full_tensor = torch.cat(all_data_tensors, dim=0)
        
        # --- IMPLEMENTACIÓN DE PURGED SPLIT CRONOLÓGICO ---
        # 1. Definir tamaño del Gap (Purga) igual al tamaño de la ventana.
        # Esto asegura que ninguna ventana del Train termine después de que empiece la primera del Val.
        window_size = self.params.get("LSTM_WINDOW_SIZE", 20)
        purge_gap = window_size
        
        n_samples = len(full_tensor)
        n_val_start = int(n_samples * (1 - val_split)) # Índice donde empieza Validación
        n_train_end = n_val_start - purge_gap          # Índice donde termina Entrenamiento (con gap)
        
        if n_train_end <= 0:
            raise ValueError(f"Dataset insuficiente para Purged Split. Samples: {n_samples}, Gap requerido: {purge_gap}")

        # 2. División estricta con Gap
        train_tensor = full_tensor[:n_train_end]
        val_tensor = full_tensor[n_val_start:]
        
        # Creamos nuevos DataLoaders internos
        batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32
        
        internal_train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=False) # Shuffle False es vital en series temporales
        internal_val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        print(f"🧠 Entrenando LSTM en {self.device}...")
        print(f"🛡️ PURGED SPLIT ACTIVO: Gap de {purge_gap} ventanas eliminado entre Train y Val.")
        print(f"📉 Train Size: {len(train_tensor)} | Val Size: {len(val_tensor)} | Purged: {purge_gap}")

        best_val_loss = float('inf')
        
        for epoch in range(self.params["LSTM_EPOCHS"]):
            # --- TRAIN LOOP ---
            self.model.train()
            train_loss = 0
            for batch in internal_train_loader:
                x_batch = batch[0] 
                
                optimizer.zero_grad()
                reconstructed, _ = self.model(x_batch)
                loss = criterion(reconstructed, x_batch)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(internal_train_loader)

            # --- VALIDATION LOOP (TEST) ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in internal_val_loader:
                    x_val = batch[0]
                    reconstructed_val, _ = self.model(x_val)
                    v_loss = criterion(reconstructed_val, x_val)
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(internal_val_loader) if len(internal_val_loader) > 0 else 0

            # Logging simple (cada 10 épocas)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.params['LSTM_EPOCHS']} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_val_loss:.6f}")
                
                # Opcional: Guardar mejor modelo basado en Test Loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Podrías guardar un checkpoint aquí si quisieras

    def encode(self, tensor_data):
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(tensor_data)
        return latent.cpu().numpy()

    def save(self, path):
        # FIX: Verificar si hay directorio antes de intentar crearlo
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()