import unittest
import numpy as np
import sys
import os
import shutil
import tempfile

# --- AJUSTE DE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(project_root)

from src.engine.context.gmm_model import RegimeDetector
from src.engine.config import CONTEXT_PARAMS

class TestGMMRegimeDetector(unittest.TestCase):
    
    def setUp(self):
        # Usamos parámetros de config pero reducimos complejidad para el test
        self.n_components = CONTEXT_PARAMS["GMM_N_COMPONENTS"]
        self.detector = RegimeDetector(n_components=self.n_components)
        
        # --- GENERACIÓN DE DATOS SINTÉTICOS CONTROLADOS ---
        np.random.seed(42)
        n_samples = 500
        
        # Simulamos 3 "Clusters" claros en un espacio 2D (más simple que el real de 10D)
        # Cluster A (Baja Volatilidad implícita): Centrado en 0
        cluster_a = np.random.normal(0, 0.5, (n_samples // 3, 2))
        vol_a = np.random.uniform(0.01, 0.02, n_samples // 3) # Vol muy baja
        
        # Cluster B (Media Volatilidad): Centrado en 5
        cluster_b = np.random.normal(5, 1.0, (n_samples // 3, 2))
        vol_b = np.random.uniform(0.05, 0.08, n_samples // 3) # Vol media
        
        # Cluster C (Alta Volatilidad): Centrado en 10
        cluster_c = np.random.normal(10, 2.0, (n_samples // 3, 2))
        vol_c = np.random.uniform(0.15, 0.25, n_samples // 3) # Vol alta
        
        # Concatenamos todo
        self.latent_vectors = np.vstack([cluster_a, cluster_b, cluster_c])
        self.anchor_metric = np.concatenate([vol_a, vol_b, vol_c])
        
        # Mezclamos los datos (Shuffle) para que no estén ordenados por defecto
        perm = np.random.permutation(len(self.latent_vectors))
        self.latent_vectors = self.latent_vectors[perm]
        self.anchor_metric = self.anchor_metric[perm]

    def test_01_fit_and_semantic_sorting(self):
        """
        Verifica que el modelo entrena y, lo más importante, 
        que el régimen 0 tiene MENOR volatilidad promedio que el régimen N.
        """
        print("\n--- Test 01: Semantic Sorting Logic ---")
        
        # Entrenamos (esto ajusta el modelo Y crea el mapa de traducción)
        self.detector.fit(self.latent_vectors, self.anchor_metric)
        
        # Predecimos sobre los mismos datos
        predicted_regimes, _ = self.detector.predict(self.latent_vectors)
        
        # Calculamos la volatilidad promedio real para cada régimen PREDICHO
        regime_vols = []
        for r in range(self.detector.model.n_components):
            mask = (predicted_regimes == r)
            if np.sum(mask) > 0:
                avg_vol = np.mean(self.anchor_metric[mask])
                regime_vols.append(avg_vol)
                print(f"Régimen {r} - Volatilidad Promedio: {avg_vol:.4f}")
            else:
                # Si un cluster se queda vacío (posible con muchos componentes y pocos datos), lo ignoramos
                pass
        
        # VERIFICACIÓN: La lista de volatilidades debe estar ordenada ascendente
        # Es decir: Vol(Regime 0) < Vol(Regime 1) < Vol(Regime 2)...
        is_sorted = all(regime_vols[i] <= regime_vols[i+1] for i in range(len(regime_vols)-1))
        
        self.assertTrue(is_sorted, 
                        f"El ordenamiento semántico falló. Las volatilidades no están en orden ascendente: {regime_vols}")

    def test_02_predict_consistency(self):
        """Verifica formato de salida de predict"""
        print("\n--- Test 02: Predict Format ---")
        self.detector.fit(self.latent_vectors, self.anchor_metric)
        
        # Tomamos 5 muestras
        sample = self.latent_vectors[:5]
        regimes, probs = self.detector.predict(sample)
        
        self.assertEqual(len(regimes), 5)
        self.assertEqual(len(probs), 5)
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1), "Las probabilidades deben estar entre 0 y 1")

    def test_03_save_load(self):
        """Verifica que el mapa de regímenes se guarda junto con el modelo"""
        print("\n--- Test 03: Save/Load System ---")
        self.detector.fit(self.latent_vectors, self.anchor_metric)
        original_map = self.detector.regime_map.copy()
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "gmm_test.joblib")
            self.detector.save(path)
            
            # Nueva instancia
            new_detector = RegimeDetector(n_components=self.n_components)
            new_detector.load(path)
            
            # Verificar que está 'fitted'
            self.assertTrue(new_detector.is_fitted)
            
            # Verificar que el mapa es idéntico
            self.assertEqual(new_detector.regime_map, original_map, 
                             "El mapa de traducción de regímenes no se cargó correctamente.")
            
            # Verificar que predice lo mismo
            pred_orig, _ = self.detector.predict(self.latent_vectors[:1])
            pred_new, _ = new_detector.predict(self.latent_vectors[:1])
            self.assertEqual(pred_orig, pred_new)

if __name__ == '__main__':
    unittest.main()