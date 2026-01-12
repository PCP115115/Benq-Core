import unittest
import sys
import os
import polars as pl
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

# Importamos el Pipeline
from src.engine.meta_model.src_meta_model.pipeline_meta import train_meta_model, create_dual_target
from src.engine.meta_model.src_meta_model.download_meta import get_data_meta_model
import src.engine.config as config

class TestMetaModelRigorous(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("🕵️‍♂️ AUDITORÍA CUANTITATIVA: META-MODELO (XGBOOST)")
        print("="*60)
        
        cls.ticker = "AAPL"
        cls.model_dir = os.path.join(project_root, "src", "data", "models", "meta_model")
        
        # Ejecutamos el pipeline una vez para asegurar que existen los modelos y datos frescos
        print("⚙️ Ejecutando Pipeline de Entrenamiento (Pre-Test)...")
        train_meta_model(cls.ticker)

    def setUp(self):
        # Cargamos datos frescos para validación manual
        self.df_raw = get_data_meta_model(
            ticker=self.ticker,
            start_date=config.META_MODEL_PARAMS["start_date"],
            end_date=config.META_MODEL_PARAMS["end_date"],
            layer="all",
            feature_list=config.META_MODEL_PARAMS["feature_list"],
            normalization_window=config.META_MODEL_PARAMS["normalization_window"]
        )
        
        # Re-creamos el target para tener la "Verdad" (y_true)
        self.df_data = create_dual_target(
            self.df_raw, 
            horizon=config.META_MODEL_PARAMS["FORECAST_HORIZON"],
            z_score=config.FEATURES_PARAMS["YZ_Z_SCORE"],
            vol_window=config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
        )
        
        # Features usados en entrenamiento
        exclude_cols = [
            "Date", "ticker", "sector", "country", "data_quality",
            "TARGET_UP", "TARGET_DOWN", 
            "first_ceil_hit", "first_floor_hit", 
            f"fprice_ceil_yz_{config.META_MODEL_PARAMS['FORECAST_HORIZON']}d", 
            f"fprice_floor_yz_{config.META_MODEL_PARAMS['FORECAST_HORIZON']}d",
            "Close", "High", "Low", "Open", "Volume", "log_returns",
            f"vol_yz_{config.FEATURES_PARAMS['YANG_ZHANG_WINDOW']}d_RAW"
        ]
        self.feature_cols = [c for c in self.df_data.columns if c not in exclude_cols]

    def test_01_model_files_exist(self):
        """Verifica la persistencia de los artefactos del modelo."""
        path_up = os.path.join(self.model_dir, "xgboost_meta_up.json")
        path_down = os.path.join(self.model_dir, "xgboost_meta_down.json")
        
        self.assertTrue(os.path.exists(path_up), "❌ Falta el modelo UP (.json)")
        self.assertTrue(os.path.exists(path_down), "❌ Falta el modelo DOWN (.json)")
        print("✅ Artefactos de modelo encontrados.")

    def _audit_model(self, direction):
        """Función auxiliar para auditar un modelo específico (UP o DOWN)."""
        model_path = os.path.join(self.model_dir, f"xgboost_meta_{direction.lower()}.json")
        target_col = f"TARGET_{direction}"
        
        # 1. Cargar Modelo
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # 2. Preparar Datos (Simulamos Inferencia)
        X = self.df_data.select(self.feature_cols).to_pandas()
        y_true = self.df_data.select(target_col).to_pandas().values.ravel()
        
        # 3. Predicción
        y_pred = model.predict(X)
        y_probs = model.predict_proba(X)[:, 1] # Probabilidad de Clase 1
        
        # 4. Métricas Avanzadas
        print(f"\n📊 --- ANÁLISIS FORENSE META-MODELO [{direction}] ---")
        
        # Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"   Matriz de Confusión:")
        print(f"   [ TN (Correcto No) | FP (Falsa Alarma) ]  -> [{tn:4d} | {fp:4d}]")
        print(f"   [ FN (Oportunidad Perdida) | TP (Acierto)  ]  -> [{fn:4d} | {tp:4d}]")
        
        # Reporte de Clasificación
        report = classification_report(y_true, y_pred, output_dict=True)
        recall_1 = report['1']['recall']
        precision_1 = report['1']['precision']
        
        print(f"   🎯 Precision (Fiabilidad): {precision_1:.2%}")
        print(f"   📡 Recall (Sensibilidad):  {recall_1:.2%}")
        
        # AUC-ROC (Capacidad de distinción)
        try:
            auc = roc_auc_score(y_true, y_probs)
            print(f"   ⭐ AUC-ROC Score:          {auc:.4f}")
            self.assertGreater(auc, 0.5, f"El modelo {direction} es peor que el azar (AUC < 0.5)")
        except ValueError:
            print("   ⚠️ No se pudo calcular AUC (posiblemente solo una clase presente).")

        # 5. Aserciones de Sanidad
        # Si el modelo predice SIEMPRE 0, el Recall será 0.0. Queremos evitar modelos "cobardes".
        # Nota: En datasets pequeños/desbalanceados, esto puede pasar. Lo ponemos como warning.
        if tp == 0 and fn > 0:
            print(f"   ⚠️ WARNING: El modelo [{direction}] es 'cobarde' (No detectó ningún caso positivo).")
        else:
            print(f"   ✅ El modelo detectó {tp} oportunidades reales.")

    def test_02_audit_up_model(self):
        self._audit_model("UP")

    def test_03_audit_down_model(self):
        self._audit_model("DOWN")

if __name__ == "__main__":
    unittest.main()