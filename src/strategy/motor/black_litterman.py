import sys
import os
import logging
import numpy as np
import polars as pl
from typing import Tuple, List, Dict
from sklearn.covariance import LedoitWolf
from datetime import datetime, date  # <--- IMPORTANTE

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from src.src_DD.loader import MarketLoader
    from src.strategy.motor.returns import get_strategy_returns
    import src.strategy.config_strategy as strat_config
except ImportError as e:
    print(f"❌ Error crítico de importación en Black-Litterman: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BL_CORE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BL_Optimizer")

class BlackLittermanModel:
    def __init__(self):
        """Inicializa BL con horizontes temporales configurables."""
        self.delta = strat_config.BLACK_LITTERMAN_CONFIG["RISK_AVERSION"]
        self.tau = strat_config.BLACK_LITTERMAN_CONFIG["TAU"]
        self.opt_horizon = strat_config.BLACK_LITTERMAN_CONFIG["OPTIMIZATION_HORIZON"]
        self.model_horizon = strat_config.META_MODEL_CONFIG["FORECAST_HORIZON"]
        self.time_scale = self.opt_horizon / self.model_horizon
        self.tickers = strat_config.TICKERS_ESTRATEGIA
        self.loader = MarketLoader(actualizar_datos=False)
        
        logger.info(f"⚙️ Configuración BL: T_Opt={self.opt_horizon}d | T_Model={self.model_horizon}d | Scale={self.time_scale:.2f}x")

    def _parse_date(self, date_input):
        """Helper para asegurar tipo date."""
        if date_input is None: return None
        if isinstance(date_input, str):
            try: return datetime.strptime(date_input, "%Y-%m-%d").date()
            except: return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S").date()
        if isinstance(date_input, datetime): return date_input.date()
        return date_input

    def _get_historical_returns(self) -> Tuple[pl.DataFrame, List[str]]:
        """Obtiene retornos logarítmicos DIARIOS manteniendo la columna Date."""
        logger.info("📉 Calculando retornos diarios base...")
        
        tickers_str = "', '".join(self.tickers)
        query = f"SELECT Date, ticker, Close FROM market WHERE ticker IN ('{tickers_str}') ORDER BY Date ASC"
        df_raw = self.loader.query(query)
        
        if df_raw.is_empty():
            raise ValueError("❌ No se obtuvieron datos históricos.")

        # Cast explícito de fecha al cargar
        df_raw = df_raw.with_columns(pl.col("Date").cast(pl.Date))

        df_pivot = df_raw.pivot(index="Date", on="ticker", values="Close", aggregate_function="first").sort("Date")
        df_pivot = df_pivot.fill_null(strategy="forward").drop_nulls()
        
        valid_tickers = [c for c in df_pivot.columns if c != "Date"]
        
        log_ret_exprs = [pl.col("Date")] + [(pl.col(t) / pl.col(t).shift(1)).log().alias(t) for t in valid_tickers]
        df_returns = df_pivot.select(log_ret_exprs).drop_nulls()
        
        return df_returns, valid_tickers

    def _estimate_covariance_scaled(self, df_returns: pl.DataFrame) -> np.ndarray:
        """Estima Sigma escalada (excluyendo la columna Date)."""
        logger.info(f"🧮 Escalando matriz de riesgos a {self.opt_horizon} días...")
        
        numeric_cols = [c for c in df_returns.columns if c != "Date"]
        X = df_returns.select(numeric_cols).to_numpy()
        
        lw = LedoitWolf()
        sigma_daily = lw.fit(X).covariance_
        sigma_scaled = sigma_daily * self.opt_horizon
        return sigma_scaled

    def _calculate_prior_equilibrium(self, sigma: np.ndarray, n_assets: int) -> np.ndarray:
        w_eq = np.ones(n_assets) / n_assets
        pi = self.delta * np.dot(sigma, w_eq)
        return pi

    def _get_views_and_uncertainty(self, valid_tickers: List[str], sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"🧠 Adaptando Views del modelo (x{self.time_scale:.2f})...")
        df_strategy = get_strategy_returns()
        
        if df_strategy.is_empty(): return None, None, None

        ticker_to_idx = {t: i for i, t in enumerate(valid_tickers)}
        P_list, Q_list, Omega_list = [], [], []
        
        for row in df_strategy.to_dicts():
            ticker = row["Ticker"]
            if ticker not in ticker_to_idx: continue
                
            idx = ticker_to_idx[ticker]
            raw_view = row["Exp_Ret_%"] / 100.0 
            scaled_view = raw_view * self.time_scale
            
            p_up, p_down = row["P_Up"], row["P_Down"]
            confidence = np.clip(max(p_up, p_down), 0.51, 0.99)
            
            sigma_asset = sigma[idx, idx]
            omega_val = sigma_asset * ((1 - confidence) / confidence) * self.tau
            
            p_vec = np.zeros(len(valid_tickers))
            p_vec[idx] = 1.0
            
            P_list.append(p_vec)
            Q_list.append(scaled_view)
            Omega_list.append(omega_val)
            
        if not Q_list: return None, None, None
            
        return np.array(P_list), np.array(Q_list), np.diag(Omega_list)

    def run_optimization(self, analysis_date=None) -> pl.DataFrame:
        """Ejecuta la optimización BL filtrando por fecha si es necesario."""
        
        # 1. Obtener datos base
        df_returns, valid_tickers = self._get_historical_returns()
        
        # 2. FILTRADO TEMPORAL ROBUSTO
        if analysis_date:
            target_date = self._parse_date(analysis_date)
            if target_date:
                df_returns = df_returns.filter(pl.col("Date").cast(pl.Date) <= target_date)
                
                if df_returns.height < 30:
                    logger.warning(f"⚠️ Pocos datos tras el filtrado de fecha. Filas: {df_returns.height}")
                    return pl.DataFrame()
        
        # 3. Calcular Sigma
        try:
            sigma = self._estimate_covariance_scaled(df_returns)
        except Exception as e:
            logger.error(f"Error calculando covarianza: {e}")
            return pl.DataFrame()
        
        # 4. Calcular Prior
        pi = self._calculate_prior_equilibrium(sigma, len(valid_tickers))
        
        # 5. Obtener Views
        P, Q, Omega = self._get_views_and_uncertainty(valid_tickers, sigma)
        
        if P is None:
            logger.info("⚠️ Sin Views. Retornando Prior.")
            mu_bl = pi
        else:
            logger.info(f"⚡ Fusionando: Prior({self.opt_horizon}d) + Views(Ajustadas)...")
            try:
                tau_sigma_inv = np.linalg.inv(self.tau * sigma)
                omega_inv = np.linalg.inv(Omega)
                
                M = np.linalg.inv(tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P))
                term_b = np.dot(tau_sigma_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
                mu_bl = np.dot(M, term_b)
            except np.linalg.LinAlgError:
                logger.error("Error de álgebra lineal. Usando Prior.")
                mu_bl = pi

        # 6. Resultados
        results = []
        for i, ticker in enumerate(valid_tickers):
            view_val = None
            if P is not None:
                for k in range(len(Q)):
                    if P[k, i] == 1.0:
                        view_val = Q[k]
                        break
            
            results.append({
                "Ticker": ticker,
                "T_Opt_Days": self.opt_horizon,
                "Prior_Eq_%": round(pi[i] * 100, 3),
                "View_Scaled_%": round(view_val * 100, 3) if view_val is not None else None,
                "BL_Post_%": round(mu_bl[i] * 100, 3)
            })
            
        df_res = pl.DataFrame(results).sort("BL_Post_%", descending=True)
        
        print("\n" + "="*80)
        print(f"🧠 RESULTADOS BLACK-LITTERMAN FLEXIBLE")
        print(f"   Horizonte Optimización: {self.opt_horizon} días")
        print("="*80)
        print(df_res)
        
        return df_res

def run_black_litterman():
    optimizer = BlackLittermanModel()
    return optimizer.run_optimization()

if __name__ == "__main__":
    run_black_litterman()