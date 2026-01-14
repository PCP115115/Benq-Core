import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
import os
import shutil
import xgboost as xgb
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from datetime import datetime, timedelta

# --- AJUSTE DE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# Imports
import src.engine.config as config
from src.engine.meta_model.src_meta_model import pipeline_meta

class TestMetaModelMetrics(unittest.TestCase):
    
    def setUp(self):
        """Setup: Directorios temporales."""
        self.temp_model_dir = os.path.join(current_dir, "temp_metrics_models")
        self.output_plots_dir = os.path.join(current_dir, "meta_model_plots")
        
        os.makedirs(self.temp_model_dir, exist_ok=True)
        os.makedirs(self.output_plots_dir, exist_ok=True)
        
        # Redirigir guardado de modelos
        pipeline_meta.MODEL_DIR = self.temp_model_dir
        
        # Configuración para el test
        config.META_MODEL_PARAMS["FORECAST_HORIZON"] = 5
        config.FEATURES_PARAMS["YZ_Z_SCORE"] = 2.0
        config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"] = 20
        # Forzamos modo test para que haga el split interno
        config.META_MODEL_PARAMS["META_MODEL_TRAIN_PARAMS"]["TEST_MODE"] = True
        config.META_MODEL_PARAMS["META_MODEL_TRAIN_PARAMS"]["TRAIN_TEST_SPLIT_RATIO"] = 0.7

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)

    def _create_smart_data(self, n_rows=1000):
        """
        Crea datos sintéticos con PATRONES para UP y DOWN.
        Soluciona el error de XGBoost asegurando que existan ambas clases.
        """
        start_date = datetime(2022, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        np.random.seed(42)
        
        # 1. Generamos Features Aleatorias
        rsi = np.random.uniform(20, 80, n_rows)
        p_trend_up = np.random.uniform(0, 1, n_rows)
        p_rev_down = np.random.uniform(0, 1, n_rows) # Usaremos esto para generar crashes
        
        # 2. Inicializamos Precios Base
        close = np.full(n_rows, 100.0)
        high = np.full(n_rows, 101.0) # Barrera superior aprox 102
        low = np.full(n_rows, 99.0)   # Barrera inferior aprox 98
        vol_raw = np.full(n_rows, 0.01) # 1% vol
        
        # 3. INYECCIÓN DE PATRONES (Lógica Causal)
        
        # --- PATRÓN ALCISTA (UP) ---
        # Si Trend_UP es alto (>0.8), el precio rompe hacia arriba
        target_up_indices = np.where(p_trend_up > 0.8)[0]
        
        # --- PATRÓN BAJISTA (DOWN) ---
        # Si Reversion_Down es alto (>0.8), el precio se desploma
        target_down_indices = np.where(p_rev_down > 0.8)[0]
        
        # Aplicamos al futuro (i+1)
        # Horizon = 5, así que modificamos el precio siguiente
        for i in range(n_rows - 5):
            # Lógica UP
            if i in target_up_indices:
                high[i+1] = 105.0 # Rompe techo (100 + 1%*2 = 102)
            
            # Lógica DOWN (Prioridad al crash si coinciden)
            if i in target_down_indices:
                low[i+1] = 90.0   # Rompe suelo (100 - 1%*2 = 98)
                high[i+1] = 100.0 # Anula subida si hay crash
        
        df = pl.DataFrame({
            "Date": dates,
            "ticker": ["METRICS_TEST"] * n_rows,
            "Close": close,
            "High": high,
            "Low": low,
            "Open": close,
            "Volume": np.random.randint(1000, 10000, n_rows),
            "log_returns": np.random.normal(0, 0.01, n_rows),
            "vol_yz_20d_RAW": vol_raw,
            
            # Features
            "rsi_14": rsi,
            "macd_line": np.random.randn(n_rows),
            "ker_10": np.random.randn(n_rows),
            "amihud_20d": np.random.randn(n_rows),
            "vol_yz_20d": np.random.randn(n_rows),
            
            # Contexto
            "market_regime": np.random.randint(0, 3, n_rows),
            "regime_probability": np.random.rand(n_rows),
            
            # Expertos
            "P_Trend_Up": p_trend_up, 
            "P_Trend_Down": np.random.rand(n_rows),
            "P_Rev_Up": np.random.rand(n_rows),
            "P_Rev_Down": p_rev_down,
            "P_Vol_Exp": np.random.rand(n_rows),
            "P_Vol_Comp": np.random.rand(n_rows)
        })
        return df

    def _plot_roc_curve(self, y_test, y_probs, direction):
        """Genera y guarda la curva ROC."""
        try:
            auc = roc_auc_score(y_test, y_probs)
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Meta Model [{direction}]')
            plt.legend(loc="lower right")
            
            path = os.path.join(self.output_plots_dir, f"roc_curve_{direction}.png")
            plt.savefig(path)
            plt.close()
            print(f"   🖼️ ROC Curve guardada: {path}")
        except Exception as e:
            print(f"⚠️ No se pudo graficar ROC para {direction}: {e}")

    def _plot_feature_importance(self, model, direction):
        """Genera gráfico de importancia de características."""
        plt.figure(figsize=(10, 6))
        # Usamos importance_type='gain' para ver qué feature aporta más información real
        xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='gain', title=f'Feature Importance (Gain) [{direction}]')
        
        path = os.path.join(self.output_plots_dir, f"feature_importance_{direction}.png")
        plt.savefig(path)
        plt.close()
        print(f"   🖼️ Feature Importance guardada: {path}")

    def _plot_confusion_matrix(self, y_test, y_pred, direction):
        """Genera matriz de confusión visual."""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix [{direction}]')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        path = os.path.join(self.output_plots_dir, f"confusion_matrix_{direction}.png")
        plt.savefig(path)
        plt.close()
        print(f"   🖼️ Matriz Confusión guardada: {path}")

    @patch('src.engine.meta_model.src_meta_model.pipeline_meta.get_data_meta_model')
    def test_comprehensive_metrics(self, mock_get_data):
        """
        Ejecuta entrenamiento, evaluación y generación de gráficos.
        """
        print("\n--- Test de Métricas y Visualización (Meta-Modelo) ---")
        
        # 1. Generar Datos
        n_rows = 3000
        mock_df = self._create_smart_data(n_rows=n_rows)
        mock_get_data.return_value = mock_df
        
        # 2. Entrenar (Ahora no fallará porque hay casos UP y DOWN)
        pipeline_meta.train_meta_model("METRICS_TEST")
        
        # 3. Cargar Modelos y Validar
        directions = ["UP", "DOWN"]
        
        # Recreamos dataset de validación
        print("   > Generando dataset de validación externa...")
        val_df = self._create_smart_data(n_rows=500)
        
        horizon = config.META_MODEL_PARAMS["FORECAST_HORIZON"]
        z_score = config.FEATURES_PARAMS["YZ_Z_SCORE"]
        vol_window = config.FEATURES_PARAMS["YANG_ZHANG_WINDOW"]
        val_df = pipeline_meta.create_dual_target(val_df, horizon, z_score, vol_window)
        
        exclude_cols = [
            "Date", "ticker", "sector", "country", "data_quality",
            "TARGET_UP", "TARGET_DOWN", 
            "first_ceil_hit", "first_floor_hit", 
            f"fprice_ceil_yz_{horizon}d", f"fprice_floor_yz_{horizon}d",
            "Close", "High", "Low", "Open", "Volume", "log_returns",
            f"vol_yz_{vol_window}d_RAW"
        ]
        feature_cols = [c for c in val_df.columns if c not in exclude_cols]
        X_val = val_df.select(feature_cols).to_pandas()

        for direction in directions:
            print(f"\n   🔍 Analizando Dirección: {direction}")
            model_path = os.path.join(self.temp_model_dir, f"xgboost_meta_{direction.lower()}.json")
            
            if not os.path.exists(model_path):
                print(f"⚠️ Modelo {direction} no encontrado. Saltando.")
                continue
                
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            target_col = f"TARGET_{direction}"
            y_val = val_df.select(target_col).to_pandas().values.ravel()
            
            # Inferencia
            y_probs = model.predict_proba(X_val)[:, 1]
            y_pred = (y_probs > 0.5).astype(int)
            
            # Métricas
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            print(f"   📊 Accuracy:  {acc:.4f}")
            print(f"   📊 Precision: {prec:.4f}")
            print(f"   📊 Recall:    {rec:.4f}")
            print(f"   📊 F1-Score:  {f1:.4f}")
            
            # Gráficos
            try:
                self._plot_roc_curve(y_val, y_probs, direction)
                self._plot_feature_importance(model, direction)
                self._plot_confusion_matrix(y_val, y_pred, direction)
            except Exception as e:
                print(f"⚠️ Error en gráficos {direction}: {e}")

if __name__ == '__main__':
    unittest.main()