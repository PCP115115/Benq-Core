import sys
import os
import logging
import joblib
import polars as pl
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core (Root)

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

        logger.info(f"📂 Cargando modelos desde: {self.model_dir}")
        
        if not os.path.exists(self.model_up_path) or not os.path.exists(self.model_down_path):
            raise FileNotFoundError(f"❌ No se encuentran los modelos en {self.model_dir}")

        # Cargar Modelo UP
        self.model_up = xgb.XGBClassifier()
        self.model_up.load_model(self.model_up_path)
        
        # Cargar Modelo DOWN
        self.model_down = xgb.XGBClassifier()
        self.model_down.load_model(self.model_down_path)
        
        # Obtener features esperadas para evitar mismatch
        booster = self.model_up.get_booster()
        self.required_features = booster.feature_names
        
        logger.info(f"✅ Modelos cargados. Features esperadas ({len(self.required_features)}): {self.required_features}")

    def get_market_snapshot(self, ticker: str) -> dict:
        """Obtiene datos y calcula estado actual del mercado."""
        
        # Pedimos historial suficiente
        start_date_window = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        try:
            # 1. Pipeline de Datos
            df = get_data_meta_model(
                ticker=ticker,
                start_date=start_date_window,
                end_date=datetime.now().strftime("%Y-%m-%d"),
                layer="all",
                feature_list=engine_config.META_MODEL_PARAMS["feature_list"],
                normalization_window=engine_config.META_MODEL_PARAMS["normalization_window"]
            )
            
            if df.is_empty():
                logger.warning(f"⚠️ {ticker}: DataFrame vacío.")
                return None

            # 2. Limpieza de NaNs
            # fill_null con strategy="forward" para rellenar huecos recientes
            df = df.fill_null(strategy="forward")
            
            # Recalculamos Volatilidad YZ para la proyección de barreras
            vol_window = self.params["VOL_WINDOW"]
            
            # Chequeo de seguridad
            needed_raw = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in needed_raw):
                logger.error(f"❌ {ticker}: Faltan columnas OHLC.")
                return None
            
            df = df.with_columns(
                indicators.get_yang_zhang_volatility(
                    "Open", "High", "Low", "Close", vol_window
                )
            )

            # Tomamos la ÚLTIMA fila
            last_row = df.tail(1)
            
            # Verificamos NaNs en features críticas
            if self.required_features:
                nulls = last_row.select(self.required_features).null_count().sum_horizontal()[0]
                if nulls > 0:
                    logger.warning(f"⚠️ {ticker}: Features contienen NaNs imposibles de rellenar. Saltando.")
                    return None

            # Extraer volatilidad calculada
            vol_col = f"vol_yz_{vol_window}d"
            if vol_col not in last_row.columns:
                 candidates = [c for c in last_row.columns if "vol_yz" in c]
                 vol_col = candidates[-1] if candidates else None
            
            if not vol_col:
                logger.error(f"❌ {ticker}: No se pudo calcular volatilidad.")
                return None
                
            return {
                "df_row": last_row,
                "price": last_row["Close"][0],
                "vol_yz": last_row[vol_col][0]
            }

        except Exception as e:
            logger.error(f"❌ Error procesando {ticker}: {e}")
            return None

    def calculate_expected_returns(self) -> pl.DataFrame:
        results = []
        logger.info("🚀 Iniciando motor de decisión...")

        for ticker in self.tickers:
            snapshot = self.get_market_snapshot(ticker)
            if not snapshot:
                continue

            df_row = snapshot["df_row"]
            price = snapshot["price"]
            vol = snapshot["vol_yz"]

            # --- 1. FEATURES (Lista Blanca) ---
            if self.required_features:
                missing = [c for c in self.required_features if c not in df_row.columns]
                if missing:
                    logger.error(f"❌ {ticker}: Faltan features: {missing}")
                    continue
                X = df_row.select(self.required_features).to_pandas()
            else:
                # Fallback simple
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
            
            # Barreras
            projection = vol * z_score * np.sqrt(horizon)
            
            # Retornos Potenciales (Aprox lineal para visualización)
            r_tp_pct = projection 
            r_sl_pct = projection 
            
            # Fórmula: E[R] = (Pup * R_TP) - (Pdown * |R_SL|)
            expected_return = (p_up * r_tp_pct) - (p_down * r_sl_pct)

            results.append({
                "Ticker": ticker,
                "Date": df_row["Date"][0].strftime("%Y-%m-%d"),
                "Price": price,
                "Vol_YZ": round(vol, 4),
                "P_Up": round(p_up, 4),
                "P_Down": round(p_down, 4),
                "R_TP_%": round(r_tp_pct * 100, 2),
                "R_SL_%": round(r_sl_pct * 100, 2),
                "Exp_Ret_%": round(expected_return * 100, 4)
            })

        df_results = pl.DataFrame(results)
        
        if not df_results.is_empty():
            df_results = df_results.sort("Exp_Ret_%", descending=True)
            print("\n" + "="*60)
            print("📊 TABLERO DE ESTRATEGIA QUANT - RESULTADOS E[R]")
            print("="*60)
            print(df_results)
        else:
            logger.warning("⚠️ No se generaron oportunidades válidas.")

        return df_results

# --- ENTRY POINT ---
def get_strategy_returns():
    engine = MetaStrategyEngine()
    engine.initialize()
    return engine.calculate_expected_returns()

if __name__ == "__main__":
    df = get_strategy_returns()
    if strat_config.OUTPUT_CONFIG["EXPORT_TO_CSV"] and not df.is_empty():
        df.write_csv(strat_config.OUTPUT_CONFIG["OUTPUT_FILENAME"])