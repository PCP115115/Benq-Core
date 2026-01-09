import joblib
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class RegimeDetector:
    def __init__(self, n_components=5, covariance_type="full"):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42,
            n_init=5 # Aumentamos intentos para mejor convergencia
        )
        self.is_fitted = False
        self.regime_map = {} # Diccionario para reordenar {Label_Viejo: Label_Nuevo}

    def fit(self, latent_vectors: np.ndarray, anchor_metric: np.ndarray):
        """
        Entrena el GMM y REORDENA los clústeres basándose en una métrica ancla (ej. Volatilidad).
        
        Args:
            latent_vectors: Embeddings del LSTM.
            anchor_metric: Array 1D con la métrica para ordenar (ej. Volatilidad real).
                           Debe estar alineado con latent_vectors.
        """
        print(f"🧩 Ajustando GMM ({self.model.n_components} componentes)...")
        self.model.fit(latent_vectors)
        
        # --- PASO CRÍTICO: Semantic Sorting ---
        # 1. Predecimos los labels 'brutos' (aleatorios)
        raw_labels = self.model.predict(latent_vectors)
        
        # 2. Calculamos la media de la métrica ancla para cada clúster
        cluster_stats = []
        for i in range(self.model.n_components):
            # Máscara booleana para los datos de este clúster
            mask = (raw_labels == i)
            if np.sum(mask) > 0:
                mean_val = np.mean(anchor_metric[mask])
            else:
                mean_val = -1.0 # Clúster vacío (raro)
            
            cluster_stats.append((i, mean_val))
        
        # 3. Ordenamos los clústeres de MENOR a MAYOR valor de la métrica
        # Ejemplo: Si usamos Volatilidad, 0 será Calma, 4 será Pánico.
        cluster_stats.sort(key=lambda x: x[1])
        
        # 4. Creamos el mapa de traducción {Raw_ID -> Sorted_ID}
        self.regime_map = {old_id: new_id for new_id, (old_id, _) in enumerate(cluster_stats)}
        
        print(f"⚓ Regímenes reordenados por métrica ancla (Map: {self.regime_map})")
        self.is_fitted = True

    def predict(self, latent_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve el régimen ORDENADO y la probabilidad."""
        if not self.is_fitted:
            raise RuntimeError("El modelo GMM no está entrenado.")
        
        # 1. Predicción cruda
        probs = self.model.predict_proba(latent_vectors)
        raw_regimes = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        
        # 2. Traducción semántica (Raw -> Ordenado)
        # Usamos vectorización de numpy para velocidad
        sorted_regimes = np.vectorize(self.regime_map.get)(raw_regimes)
        
        return sorted_regimes, max_probs

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Guardamos el modelo Y el mapa de traducción
        joblib.dump({'model': self.model, 'map': self.regime_map}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.regime_map = data['map']
        self.is_fitted = True