import sys
import os
import logging
import polars as pl
import numpy as np
import xgboost as xgb
# CORRECCIÓN 1: Añadimos 'date' a los imports
from datetime import datetime, timedelta, date

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    import src.engine.config as engine_config
    import src.strategy.config_strategy as strat_config
    import src.engine.src_features.indicators as indicators
    from src.engine.meta_model.src_meta_model.download_meta import get_data_meta_model
    from src.engine.meta_model.src_meta_model.pipeline_meta import train_meta_model
except ImportError as e:
    print(f"❌ Error crítico de importación en Strategy Engine: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [STRATEGY_CORE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReturnEngine")

class MetaStrategyEngine:
    def __init__(self):
        self.tickers = strat_config.TICKERS_ESTRATEGIA
        self.params = strat_config.META_MODEL_CONFIG
        
        # Ruta dinámica a los modelos
        self.model_dir = os.path.join(src_dir, "data", "models", "meta_model")
        self.model_up_path = os.path.join(self.model_dir, "xgboost_meta_up.json")
        self.model_down_path = os.path.join(self.model_dir, "xgboost_meta_down.json")
        
        self.model_up = None
        self.model_down = None
        self.required_features = []

    def initialize(self):
        """Carga modelos y determina qué features son necesarias."""
        if self.params["FORCE_RETRAIN"]:
            logger.info("⚠️ FORCE_RETRAIN activado. Ejecutando entrenamiento...")
            train_meta_model() 

        if not os.path.exists(self.model_up_path) or not os.path.exists(self.model_down_path):
            logger.warning(f"⚠️ No se encuentran modelos en {self.model_dir}. Intentando entrenar...")
            train_meta_model()

        # Cargar Modelo UP
        self.model_up = xgb.XGBClassifier()
        self.model_up.load_model(self.model_up_path)
        
        # Cargar Modelo DOWN
        self.model_down = xgb.XGBClassifier()
        self.model_down.load_model(self.model_down_path)
        
        # Obtener features esperadas
        booster = self.model_up.get_booster()
        self.required_features = booster.feature_names

    def get_market_snapshot(self, ticker: str, analysis_date: str = None) -> dict:
        """
        Obtiene datos y calcula estado actual del mercado.
        """
        
        # Determinar fecha de corte (Fin de ventana)
        if analysis_date:
            end_date_str = analysis_date
            try:
                dt_anchor = datetime.strptime(analysis_date, "%Y-%m-%d")
            except ValueError:
                dt_anchor = datetime.strptime(analysis_date.split(" ")[0], "%Y-%m-%d")
                end_date_str = dt_anchor.strftime("%Y-%m-%d")
        else:
            dt_anchor = datetime.now()
            end_date_str = dt_anchor.strftime("%Y-%m-%d")

        # Fecha de inicio (Ventana histórica necesaria para indicadores)
        start_date_window = (dt_anchor - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            # 1. Pipeline de Datos
            df = get_data_meta_model(
                ticker=ticker,
                start_date=start_date_window,
                end_date=end_date_str, 
                layer="all",
                feature_list=engine_config.META_MODEL_PARAMS["feature_list"],
                normalization_window=engine_config.META_MODEL_PARAMS["normalization_window"]
            )
            
            if df.is_empty():
                return None

            # 2. Limpieza de NaNs (CORRECCIÓN 2: Limpieza más agresiva para TSLA)
            df = df.fill_null(strategy="forward").drop_nulls()
            
            if df.is_empty():
                logger.warning(f"⚠️ {ticker}: DataFrame vacío tras limpieza de nulos.")
                return None

            # Recalculamos Volatilidad YZ
            vol_window = self.params["VOL_WINDOW"]
            needed_raw = ["Open", "High", "Low", "Close"]
            
            if not all(col in df.columns for col in needed_raw):
                return None
            
            df = df.with_columns(
                indicators.get_yang_zhang_volatility(
                    "Open", "High", "Low", "Close", vol_window
                )
            )

            # Tomamos la ÚLTIMA fila disponible
            last_row = df.tail(1)
            
            # Verificamos si la fecha de la última fila es relevante
            last_date = last_row["Date"][0]
            
            # CORRECCIÓN 1: Aquí es donde fallaba 'date'
            if isinstance(last_date, (datetime, date)):
                # Convertir a date si es datetime para la resta
                d1 = dt_anchor.date() if isinstance(dt_anchor, datetime) else dt_anchor
                d2 = last_date.date() if isinstance(last_date, datetime) else last_date
                days_diff = (d1 - d2).days
            else:
                days_diff = 0
            
            if days_diff > 10:
                logger.warning(f"⚠️ {ticker}: Datos obsoletos ({last_date} vs {end_date_str}). Diff: {days_diff} días")
                return None

            # Verificamos NaNs en features críticas
            if self.required_features:
                # Filtrar solo las que existen en el DF (por seguridad)
                valid_feats = [f for f in self.required_features if f in last_row.columns]
                nulls = last_row.select(valid_feats).null_count().sum_horizontal()[0]
                if nulls > 0:
                    return None

            # Extraer volatilidad
            vol_col = f"vol_yz_{vol_window}d"
            if vol_col not in last_row.columns:
                 candidates = [c for c in last_row.columns if "vol_yz" in c]
                 vol_col = candidates[-1] if candidates else None
            
            if not vol_col:
                return None
                
            return {
                "df_row": last_row,
                "price": last_row["Close"][0],
                "vol_yz": last_row[vol_col][0]
            }

        except Exception as e:
            logger.error(f"❌ Error procesando {ticker}: {e}")
            return None

    def calculate_expected_returns(self, analysis_date: str = None) -> pl.DataFrame:
        """Calcula retornos esperados."""
        results = []
        
        for ticker in self.tickers:
            snapshot = self.get_market_snapshot(ticker, analysis_date=analysis_date)
            
            if not snapshot:
                continue

            df_row = snapshot["df_row"]
            price = snapshot["price"]
            vol = snapshot["vol_yz"]

            # --- 1. FEATURES ---
            if self.required_features:
                # Rellenar con 0 si falta alguna columna (parche de seguridad)
                missing = [c for c in self.required_features if c not in df_row.columns]
                if missing:
                    logger.warning(f"⚠️ {ticker} falta features: {missing}")
                    continue
                    
                X = df_row.select(self.required_features).to_pandas()
            else:
                exclude = ["Date", "ticker", "sector", "country", "Close", "High", "Low", "Open", "Volume"]
                cols = [c for c in df_row.columns if c not in exclude]
                X = df_row.select(cols).to_pandas()

            # --- 2. INFERENCIA ---
            try:
                p_up = self.model_up.predict_proba(X)[0][1]
                p_down = self.model_down.predict_proba(X)[0][1]
            except Exception as e:
                logger.error(f"❌ Error inferencia {ticker}: {e}")
                continue

            # --- 3. CÁLCULO E[R] ---
            horizon = self.params["FORECAST_HORIZON"]
            z_score = self.params["YZ_Z_SCORE"]
            
            projection = vol * z_score * np.sqrt(horizon)
            
            r_tp_pct = projection 
            r_sl_pct = projection 
            
            expected_return = (p_up * r_tp_pct) - (p_down * r_sl_pct)

            results.append({
                "Ticker": ticker,
                "Date": df_row["Date"][0].strftime("%Y-%m-%d"),
                "Price": price,
                "Vol_YZ": round(vol, 4),
                "P_Up": round(p_up, 4),
                "P_Down": round(p_down, 4),
                "Exp_Ret_%": round(expected_return * 100, 4)
            })

        df_results = pl.DataFrame(results)
        
        if df_results.is_empty():
            logger.warning(f"⚠️ No se generaron señales para {analysis_date}. Revisa datos históricos.")
        
        return df_results

# --- ENTRY POINT COMPATIBLE ---
def get_strategy_returns(analysis_date=None):
    engine = MetaStrategyEngine()
    engine.initialize()
    if hasattr(analysis_date, 'strftime'):
        analysis_date = analysis_date.strftime("%Y-%m-%d")
        
    return engine.calculate_expected_returns(analysis_date=analysis_date)

if __name__ == "__main__":
    df = get_strategy_returns()
    print(df)