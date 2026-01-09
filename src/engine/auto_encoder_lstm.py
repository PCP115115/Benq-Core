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
        
        return torch.FloatTensor(X).to(self.device), data_arr

    def fit(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["LSTM_LR"])
        criterion = nn.MSELoss()
        self.model.train()
        
        print(f"🧠 Entrenando LSTM en {self.device}...")
        for epoch in range(self.params["LSTM_EPOCHS"]):
            total_loss = 0
            for batch in train_loader:
                x_batch = batch[0] 
                
                optimizer.zero_grad()
                reconstructed, _ = self.model(x_batch)
                loss = criterion(reconstructed, x_batch)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Reduce log verbosity for tests
            if (epoch+1) % 10 == 0:
                pass 

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