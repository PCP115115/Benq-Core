import sys
import os
import logging
import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime, timedelta

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(strategy_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    import src.strategy.config_strategy as strat_config
    import src.engine.config as engine_config
    import src.engine.src_features.indicators as indicators
    from src.engine.meta_model.src_meta_model.download_meta import get_data_meta_model
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ACCURACY] - %(message)s')
logger = logging.getLogger("AccuracyTester")

class ReturnAccuracyTester:
    def __init__(self, lookback_days=120):
        self.tickers = strat_config.TICKERS_ESTRATEGIA
        self.horizon = strat_config.META_MODEL_CONFIG["FORECAST_HORIZON"]
        self.lookback = lookback_days
        
        self.model_dir = os.path.join(src_dir, "data", "models", "meta_model")
        self.model_up = xgb.XGBClassifier()
        self.model_down = xgb.XGBClassifier()
        
        self._load_models()

    def _load_models(self):
        try:
            self.model_up.load_model(os.path.join(self.model_dir, "xgboost_meta_up.json"))
            self.model_down.load_model(os.path.join(self.model_dir, "xgboost_meta_down.json"))
            self.features = self.model_up.get_booster().feature_names
            logger.info(f"✅ Modelos cargados. Features: {len(self.features)}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            sys.exit(1)

    def get_test_data(self, ticker):
        """Pipeline protegido contra errores de datos."""
        start_date = (datetime.now() - timedelta(days=self.lookback + 250)).strftime("%Y-%m-%d")
        
        try:
            # 1. Obtener Features
            df = get_data_meta_model(
                ticker=ticker,
                start_date=start_date,
                end_date=datetime.now().strftime("%Y-%m-%d"),
                layer="all",
                feature_list=engine_config.META_MODEL_PARAMS["feature_list"],
                normalization_window=engine_config.META_MODEL_PARAMS["normalization_window"]
            )
            
            if df.is_empty(): return None

            # 2. Limpieza Previa (Critical Fix para GMM/XGBoost)
            df = df.fill_null(strategy="forward")
            
            # 3. Calcular Target y Volatilidad
            vol_window = strat_config.META_MODEL_CONFIG["VOL_WINDOW"]
            df = df.with_columns(
                indicators.get_yang_zhang_volatility("Open", "High", "Low", "Close", vol_window)
            )
            
            df = df.with_columns(
                ((pl.col("Close").shift(-self.horizon) - pl.col("Close")) / pl.col("Close") * 100).alias("Actual_Return_Pct")
            )
            
            # 4. Generar Predicciones
            valid_cols = [c for c in self.features if c in df.columns]
            if not valid_cols: return None
            
            # Limpiar filas con nulos EN LAS FEATURES antes de predecir
            # Esto evita el error de "Input X contains NaN"
            df_pred = df.drop_nulls(subset=valid_cols)
            
            if df_pred.height == 0: return None

            X = df_pred.select(valid_cols).to_pandas()
            
            prob_up = self.model_up.predict_proba(X)[:, 1]
            prob_down = self.model_down.predict_proba(X)[:, 1]
            
            df_pred = df_pred.with_columns([
                pl.Series("Prob_Up", prob_up),
                pl.Series("Prob_Down", prob_down)
            ])
            
            # 5. Calcular E[R]
            z_score = strat_config.META_MODEL_CONFIG["YZ_Z_SCORE"]
            t_factor = np.sqrt(self.horizon)
            
            vol_col = f"vol_yz_{vol_window}d"
            if vol_col not in df_pred.columns: 
                 candidates = [c for c in df_pred.columns if "vol_yz" in c]
                 vol_col = candidates[0] if candidates else None
            
            if not vol_col: return None

            df_pred = df_pred.with_columns(
                (pl.col(vol_col) * z_score * t_factor * (pl.col("Prob_Up") - pl.col("Prob_Down")) * 100).alias("Predicted_Raw_Pct")
            )
            
            # Limpiar filas donde falte el target (futuro)
            df_final = df_pred.drop_nulls(subset=["Actual_Return_Pct", "Predicted_Raw_Pct"])
            
            return df_final.tail(self.lookback)

        except Exception as e:
            logger.warning(f"⚠️ Error procesando {ticker}: {e}")
            return None

    def run_analysis(self):
        results = {}
        print("\n" + "="*60)
        print(f"📊 TEST DE PRECISIÓN (Horizonte: {self.horizon} días | Muestras: {self.lookback})")
        print("="*60)
        
        for ticker in self.tickers:
            print(f"🔹 Analizando {ticker}...")
            df = self.get_test_data(ticker)
            
            if df is None or df.height < 10:
                print(f"   ⏭️ Saltando {ticker} (Datos insuficientes o error).")
                continue
                
            y_true = df["Actual_Return_Pct"].to_numpy()
            y_pred = df["Predicted_Raw_Pct"].to_numpy()
            
            # Métricas
            ic, _ = pearsonr(y_pred, y_true)
            rank_ic, _ = spearmanr(y_pred, y_true)
            
            # Hit Rate (ignorando ceros exactos)
            nonzero = y_pred != 0
            if np.sum(nonzero) > 0:
                hit_rate = np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))
            else:
                hit_rate = 0.5
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            results[ticker] = {
                "IC (Pearson)": ic,
                "Rank IC": rank_ic,
                "Hit Rate (%)": hit_rate * 100,
                "RMSE": rmse,
                "Data": (df["Date"].to_numpy(), y_true, y_pred)
            }
            
            print(f"   🎯 Hit Rate: {hit_rate*100:.1f}%")
            print(f"   📉 IC Correlation: {ic:.4f}")
            
        return results

    def plot_results(self, results):
        if not results: 
            print("⚠️ No hay resultados para graficar.")
            return

        n_tickers = len(results)
        fig, axes = plt.subplots(n_tickers, 2, figsize=(14, 4 * n_tickers))
        if n_tickers == 1: axes = np.array([axes])
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Convertir a lista de listas para iteración segura
        if len(axes.shape) == 1: axes = axes.reshape(1, -1) if n_tickers==1 else axes

        for i, (ticker, stats) in enumerate(results.items()):
            dates, y_true, y_pred = stats["Data"]
            
            # AX 1: Serie Temporal
            ax1 = axes[i, 0]
            ax1.plot(dates, y_true, label="Real (Mercado)", color='gray', alpha=0.4)
            ax1.plot(dates, y_pred, label="Modelo E[R]", color='blue', linewidth=1.2)
            ax1.set_title(f"{ticker} - RMSE: {stats['RMSE']:.2f}")
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.3)
            
            # AX 2: Scatter
            ax2 = axes[i, 1]
            ax2.scatter(y_pred, y_true, alpha=0.5, c='purple', s=15)
            
            # Regresión
            if len(y_pred) > 1:
                m, b = np.polyfit(y_pred, y_true, 1)
                ax2.plot(y_pred, m*y_pred + b, color='red', lw=1, 
                         label=f"IC: {stats['IC (Pearson)']:.2f}")
            
            ax2.axhline(0, color='k', lw=0.5); ax2.axvline(0, color='k', lw=0.5)
            ax2.set_title(f"Correlación (Hit Rate: {stats['Hit Rate (%)']:.1f}%)")
            ax2.set_xlabel("Predicción"); ax2.set_ylabel("Realidad")
            ax2.legend()
            ax2.grid(alpha=0.3)

        print("\n📈 Generando visualización...")
        plt.show()

if __name__ == "__main__":
    tester = ReturnAccuracyTester(lookback_days=180)
    results = tester.run_analysis()
    tester.plot_results(results)