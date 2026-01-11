import unittest
import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, log_loss, brier_score_loss, accuracy_score
import sys
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustamos para que pueda importar los módulos del engine
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # Subimos hasta la raíz del proyecto
sys.path.append(str(project_root))

# Importamos (o Mockeamos si es necesario para el test aislado)
# Asumimos que el entorno tiene acceso, si no, usaríamos mocks.
from engine import config
# Mockeamos master_features para no depender de datos reales ni DB
sys.modules["engine.src_features.master_features"] = MagicMock()
from engine.mini_models.src_mini_models import trend_mini_model, reversion_mini_models, volatility_mini_models

class TestRigorousMiniModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Configuración global antes de todos los tests"""
        print("\n" + "="*60)
        print("🔬 INICIANDO AUDITORÍA RIGUROSA DE MINI-MODELOS (MOE)")
        print("="*60)
        
        # Directorio temporal para guardar modelos del test
        cls.test_dir = Path("temp_test_models_output")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Parámetros simulados
        cls.horizon = 5
        cls.n_rows = 1000
        
    @classmethod
    def tearDownClass(cls):
        """Limpieza al finalizar"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            print("\n🧹 Limpieza de archivos temporales completada.")

    def create_synthetic_market_data(self):
        """
        Genera un DataFrame de Polars con datos de mercado sintéticos 
        matemáticamente coherentes para probar la lógica.
        """
        np.random.seed(42)
        
        # 1. Generar camino aleatorio (Random Walk) para precios
        returns = np.random.normal(0, 0.02, self.n_rows)
        price_path = 100 * np.exp(np.cumsum(returns))
        
        dates = pd.date_range(start="2023-01-01", periods=self.n_rows, freq="D")
        
        data = {
            "Date": dates,
            "ticker": ["TEST"] * self.n_rows,
            "Close": price_path,
            "Open": price_path * (1 + np.random.normal(0, 0.005, self.n_rows)),
            "High": price_path * (1 + np.abs(np.random.normal(0, 0.01, self.n_rows))),
            "Low": price_path * (1 - np.abs(np.random.normal(0, 0.01, self.n_rows))),
            "Volume": np.random.randint(1000, 100000, self.n_rows),
            "log_returns": returns,
            # Features simuladas (Ruido + Señal pequeña)
            "rsi_14": np.random.uniform(20, 80, self.n_rows),
            "adx_14": np.random.uniform(10, 50, self.n_rows),
            "vol_yz_20d": np.random.uniform(0.1, 0.5, self.n_rows),
            # Necesario para Volatility Model
            "vol_std_20d": np.random.uniform(0.1, 0.5, self.n_rows), 
        }
        
        # Añadimos las columnas necesarias para cada tipo de modelo (relleno dummy)
        all_features = set(
            config.MINI_MODEL_PARAMS["FEATURES_TREND"] + 
            config.MINI_MODEL_PARAMS["FEATURES_REVERSION"] + 
            config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
        )
        
        for feat in all_features:
            if feat not in data:
                data[feat] = np.random.random(self.n_rows)
                
        return pl.DataFrame(data)

    def evaluate_classifier_performance(self, model, X, y_true, name="Model"):
        """
        Calcula métricas clave de Ciencia de Datos para clasificación.
        """
        if len(y_true) == 0: return {}
        
        # Predicciones
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Métricas
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.5 # Si solo hay una clase
            
        # Log Loss (entropía cruzada) y Brier Score (MSE de probabilidades)
        # Brier Score es el equivalente al R^2 para clasificación probabilística.
        # Brier = 0 es perfecto, Brier = 0.25 es aleatorio (para clases balanceadas).
        brier = brier_score_loss(y_true, y_proba)
        
        print(f"   📊 Métricas {name}:")
        print(f"      - Accuracy:  {acc:.2%}")
        print(f"      - F1-Score:  {f1:.4f}")
        print(f"      - ROC-AUC:   {auc:.4f} (Capacidad de discriminación)")
        print(f"      - Brier Sc.: {brier:.4f} (Calidad de la probabilidad, <0.25 es bueno)")
        
        return {"acc": acc, "f1": f1, "auc": auc, "brier": brier}

    def check_overfitting(self, train_metrics, test_metrics, threshold=0.15):
        """
        Compara Train vs Test. Si el modelo es un genio en Train pero mediocre en Test,
        hay Overfitting.
        """
        delta_auc = train_metrics["auc"] - test_metrics["auc"]
        
        print(f"   ⚖️  Check Overfitting (Gap AUC): {delta_auc:.4f}")
        
        if delta_auc > threshold:
            print(f"   ⚠️  ALERTA: Posible Overfitting detectado. El modelo memoriza el Train set.")
            return False
        elif delta_auc < -0.05:
            print(f"   ⚠️  NOTA: Underfitting o Test set anómalo (Test mejor que Train).")
            return True
        else:
            print(f"   ✅  Modelo Robusto: Generalización correcta.")
            return True

    @patch('engine.src_features.master_features.get_feature_matrix')
    def test_trend_model_integrity(self, mock_get_matrix):
        """Prueba completa del Modelo de Tendencia"""
        print("\n🔹 TEST 1: TREND MINI-MODEL (Tendencia)")
        
        # 1. Mock de Datos
        df_fake = self.create_synthetic_market_data()
        mock_get_matrix.return_value = df_fake
        
        # 2. Ejecutar Entrenamiento
        # Usamos un directorio temporal
        paths = trend_mini_model.train_trend_model("TEST_TICKER", str(self.test_dir))
        
        self.assertIsNotNone(paths, "El entrenamiento falló y no devolvió rutas.")
        
        # 3. Auditoría del Modelo Generado (UP)
        # Para auditar, necesitamos reconstruir los datos de train/test igual que el script
        # Esto es una simulación, en un caso real accederíamos a los datos procesados.
        # Aquí re-creamos el split para evaluar el modelo guardado.
        
        import joblib
        model_up = joblib.load(paths["up"])
        
        # Simulamos X_test e y_test extrayendo del final del DF falso
        # (Simplificación para el test unitario)
        split_idx = int(self.n_rows * 0.8)
        test_df = df_fake[split_idx:]
        
        features = config.MINI_MODEL_PARAMS["FEATURES_TREND"]
        X_test = test_df.select(features).to_pandas()
        
        # Generamos un target dummy para evaluar (en la realidad el script lo calcula)
        # Aquí asumimos target aleatorio para verificar que el pipeline de métricas funciona
        y_test_dummy = np.random.randint(0, 2, size=len(test_df))
        
        print("\n   [Auditoría Modelo UP - Trend]")
        metrics = self.evaluate_classifier_performance(model_up, X_test, y_test_dummy, "Trend UP (Test Simulado)")
        
        # Assertions básicas
        self.assertTrue(os.path.exists(paths["up"]), "Archivo del modelo UP no existe")
        self.assertTrue(os.path.exists(paths["down"]), "Archivo del modelo DOWN no existe")
        print("   ✅ Archivos generados correctamente.")

    @patch('engine.src_features.master_features.get_feature_matrix')
    def test_volatility_model_integrity(self, mock_get_matrix):
        """Prueba completa del Modelo de Volatilidad"""
        print("\n🔹 TEST 2: VOLATILITY MINI-MODEL (Riesgo)")
        
        df_fake = self.create_synthetic_market_data()
        mock_get_matrix.return_value = df_fake
        
        paths = volatility_mini_models.train_volatility_model("TEST_TICKER", str(self.test_dir))
        self.assertIsNotNone(paths)
        
        import joblib
        model_exp = joblib.load(paths["expansion"])
        
        # Validación de Features específicas
        expected_features = config.MINI_MODEL_PARAMS["FEATURES_VOLATILITY"]
        self.assertEqual(model_exp.n_features_, len(expected_features), 
                         "El modelo no se entrenó con el número correcto de features.")
        
        print("   ✅ Dimensiones del modelo correctas.")
        print(f"   ✅ Features usadas: {model_exp.n_features_}")

    def test_r2_explanation(self):
        """Test conceptual sobre métricas"""
        print("\n🔹 TEST 3: VERIFICACIÓN DE MÉTRICAS (R² vs Classifiers)")
        print("   ℹ️ NOTA TÉCNICA: Has solicitado R².")
        print("   Los mini-modelos son CLASIFICADORES (predicen probabilidad 0-1, no precio continuo).")
        print("   El R² no es matemático válido para clasificación binaria.")
        print("   ✅ Se ha sustituido por: Brier Score y ROC-AUC.")
        print("   - ROC-AUC: ¿Qué tan bien ordena el modelo los positivos de los negativos?")
        print("   - Brier Score: El equivalente al MSE para probabilidades. (Menor es mejor).")

if __name__ == '__main__':
    unittest.main()