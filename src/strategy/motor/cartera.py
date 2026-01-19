import sys
import os
import logging
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime, date, timedelta
from functools import reduce

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
import src.strategy.config_strategy as strat_config
from src.src_DD.loader import MarketLoader

# Motor de Retornos (Script anterior)
try:
    from src.strategy.motor.returns import get_strategy_returns
except ImportError as e:
    print(f"❌ Error crítico: No se encuentra el script de retornos (src/strategy/motor/returns.py). {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PORTFOLIO_OPT] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PortfolioOptimizer")

class PortfolioOptimizer:
    def __init__(self):
        """
        Motor de Optimización Media-Varianza con Shrinkage Ledoit-Wolf.
        """
        self.tickers = strat_config.TICKERS_ESTRATEGIA
        self.pf_config = strat_config.PORTFOLIO_CONFIG
        self.bl_config = strat_config.BLACK_LITTERMAN_CONFIG 
        
        # Loader en modo "lazy" (no actualiza, solo lee)
        self.loader = MarketLoader(actualizar_datos=False)

    def _parse_date(self, date_input):
        """Normaliza la fecha a datetime.date."""
        if date_input is None: return date.today()
        if isinstance(date_input, str):
            try: return datetime.strptime(date_input, "%Y-%m-%d").date()
            except: return datetime.strptime(date_input.split(" ")[0], "%Y-%m-%d").date()
        if isinstance(date_input, datetime): return date_input.date()
        return date_input

    def _get_historical_returns(self, end_date_dt, lookback_days=365):
        """
        Obtiene los retornos logarítmicos históricos para la matriz de covarianza.
        """
        start_date = end_date_dt - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date_dt.strftime("%Y-%m-%d")
        
        data_frames = []
        
        # [Image of Portfolio Optimization Data Alignment]
        
        for ticker in self.tickers:
            ticker_clean = ticker.upper()
            
            # --- CONSULTA SQL ROBUSTA ---
            # 1. Filtramos solo por ticker primero para ver si existe
            # 2. Usamos CAST(Date AS DATE) para asegurar compatibilidad String vs Timestamp
            sql_query = f"""
                SELECT Date, Close 
                FROM market 
                WHERE ticker = '{ticker_clean}' 
                AND CAST(Date AS DATE) >= CAST('{start_str}' AS DATE)
                AND CAST(Date AS DATE) <= CAST('{end_str}' AS DATE)
                ORDER BY Date
            """
            
            df = self.loader.query(sql_query)
            
            if df is not None and not df.is_empty():
                # Polars: Asegurar tipos
                df_clean = df.select([
                    pl.col("Date").cast(pl.Date),
                    pl.col("Close").cast(pl.Float64).alias(ticker)
                ])
                
                # Validar longitud mínima
                if df_clean.height > 10:
                    data_frames.append(df_clean)
                else:
                    logger.warning(f"⚠️ {ticker}: Datos insuficientes ({df_clean.height} filas) en rango {start_str} - {end_str}")
            else:
                # Si falla, puede que el ticker no tenga datos en ese rango
                # logger.warning(f"⚠️ {ticker}: Query SQL retornó 0 filas.")
                pass
        
        if not data_frames:
            logger.error(f"❌ No se recuperaron datos históricos para ningún activo entre {start_str} y {end_str}.")
            return None

        try:
            # Join Inner para asegurar integridad de datos (intersección de fechas)
            df_merged = reduce(lambda left, right: left.join(right, on="Date", how="inner"), data_frames)
        except Exception as e:
            logger.error(f"Error uniendo históricos: {e}")
            return None

        # Ordenar y calcular retornos logarítmicos
        df_merged = df_merged.sort("Date")
        
        # Necesitamos al menos 30 días de historia conjunta
        if df_merged.height < 30:
             logger.error(f"❌ Historia conjunta insuficiente ({df_merged.height} filas).")
             return None

        cols = [c for c in df_merged.columns if c != "Date"]
        
        df_returns = df_merged.select(
            [pl.col(c).log().diff().alias(c) for c in cols]
        ).drop_nulls()
        
        return df_returns

    def _estimate_covariance_shrinkage(self, df_returns):
        """Calcula covarianza con Shrinkage."""
        X = df_returns.to_numpy()
        horizon_days = self.bl_config.get("OPTIMIZATION_HORIZON", 5)

        try:
            lw = LedoitWolf()
            shrunk_cov_daily = lw.fit(X).covariance_
            sigma_period = shrunk_cov_daily * horizon_days
            return sigma_period
        except Exception as e:
            logger.warning(f"⚠️ Fallo en Shrinkage ({e}). Usando covarianza estándar.")
            return np.cov(X, rowvar=False) * horizon_days

    def _get_optimization_inputs(self, analysis_date=None):
        """Orquesta E[R] y Sigma."""
        target_date = self._parse_date(analysis_date)
        date_str = target_date.strftime("%Y-%m-%d")

        # A. E[R] (Predicciones)
        df_preds = get_strategy_returns(analysis_date=date_str)
        if df_preds is None or df_preds.is_empty():
            logger.warning(f"⚠️ Sin predicciones para {date_str}.")
            return None, None, None

        df_preds = df_preds.sort("Ticker")
        pred_tickers = df_preds["Ticker"].to_list()
        mu = df_preds["Exp_Ret_%"].to_numpy() / 100.0

        # B. Sigma (Histórico)
        df_hist = self._get_historical_returns(target_date)
        
        if df_hist is None:
            return None, None, None

        # C. Alineación
        hist_tickers = df_hist.columns
        common_tickers = sorted(list(set(pred_tickers) & set(hist_tickers)))
        
        if not common_tickers:
            logger.error("❌ No hay coincidencia entre predicciones y datos históricos.")
            return None, None, None

        # Filtrar Mu
        idx_mu = [pred_tickers.index(t) for t in common_tickers]
        mu_aligned = mu[idx_mu]

        # Filtrar Sigma
        df_hist_aligned = df_hist.select(common_tickers)
        sigma_aligned = self._estimate_covariance_shrinkage(df_hist_aligned)

        return common_tickers, mu_aligned, sigma_aligned

    def _calculate_metrics(self, weights, mu, sigma):
        p_ret = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return p_ret, p_vol

    # --- FUNCIONES OBJETIVO ---
    def _obj_neg_sharpe(self, weights, mu, sigma, rf_period):
        ret, vol = self._calculate_metrics(weights, mu, sigma)
        if vol < 1e-6: return 1e6
        return -((ret - rf_period) / vol)

    def _obj_neg_return(self, weights, mu, sigma):
        ret, _ = self._calculate_metrics(weights, mu, sigma)
        return -ret

    def _obj_min_volatility(self, weights, mu, sigma):
        _, vol = self._calculate_metrics(weights, mu, sigma)
        return vol

    def optimize(self, analysis_date=None) -> pl.DataFrame:
        tickers, mu, sigma = self._get_optimization_inputs(analysis_date)
        if tickers is None: return pl.DataFrame()

        n_assets = len(tickers)
        objective = self.pf_config["OBJECTIVE"]
        allow_shorts = self.pf_config["ALLOW_SHORTS"]
        min_w_cfg = self.pf_config["MIN_WEIGHT_PER_ASSET"]
        max_w = self.pf_config["MAX_WEIGHT_PER_ASSET"]
        
        rf_annual = self.pf_config["RISK_FREE_RATE_ANNUAL"]
        horizon_days = self.bl_config.get("OPTIMIZATION_HORIZON", 5)
        rf_period = rf_annual * (horizon_days / 365.0)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if allow_shorts:
            bounds = tuple((min_w_cfg, max_w) for _ in range(n_assets))
        else:
            lower_bound = max(0.0, min_w_cfg)
            bounds = tuple((lower_bound, max_w) for _ in range(n_assets))

        init_guess = np.array([1.0 / n_assets] * n_assets)

        if objective == "MAX_SHARPE":
            fun = self._obj_neg_sharpe
            args = (mu, sigma, rf_period)
        elif objective == "MAX_RETURN":
            fun = self._obj_neg_return
            args = (mu, sigma)
        elif objective == "MIN_VOLATILITY":
            fun = self._obj_min_volatility
            args = (mu, sigma)
        else:
            return pl.DataFrame()

        try:
            opt_result = minimize(
                fun, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-8
            )
        except Exception as e:
            logger.error(f"❌ Error en SLSQP: {e}")
            return pl.DataFrame()

        final_weights = opt_result.x
        final_weights[np.abs(final_weights) < 0.0001] = 0.0
        
        if np.sum(final_weights) != 0:
            final_weights = final_weights / np.sum(final_weights)

        results = []
        for i, t in enumerate(tickers):
            w = final_weights[i]
            results.append({
                "Ticker": t,
                "Weight": round(w, 4),
                "Weight_%": round(w * 100, 2),
                "Exp_Ret_5d_%": round(mu[i] * 100, 2)
            })

        return pl.DataFrame(results).sort("Weight", descending=True)

def run_portfolio_optimization(analysis_date=None):
    optimizer = PortfolioOptimizer()
    return optimizer.optimize(analysis_date=analysis_date)

if __name__ == "__main__":
    print(run_portfolio_optimization())