import sys
import os
import logging
import numpy as np
import polars as pl
from scipy.optimize import minimize
from datetime import datetime, date  # <--- IMPORTANTE

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS DIRECTOS ---
import src.strategy.config_strategy as strat_config
from src.src_DD.loader import MarketLoader
from src.strategy.motor.black_litterman import BlackLittermanModel

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PORTFOLIO] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PortfolioOpt")

class PortfolioOptimizer:
    def __init__(self):
        """Motor de Optimización Media-Varianza (Markowitz)."""
        self.params = strat_config.PORTFOLIO_CONFIG
        self.bl_config = strat_config.BLACK_LITTERMAN_CONFIG
        self.bl_model = BlackLittermanModel()
        self.loader = MarketLoader(actualizar_datos=False)

    def _parse_date(self, date_input):
        """Convierte cualquier entrada de fecha a datetime.date seguro."""
        if date_input is None:
            return None
        if isinstance(date_input, str):
            # Intenta formato estándar YYYY-MM-DD
            try:
                return datetime.strptime(date_input, "%Y-%m-%d").date()
            except ValueError:
                # Si falla, intenta con timestamp completo
                try:
                    return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S").date()
                except:
                    return None
        if isinstance(date_input, datetime):
            return date_input.date()
        if isinstance(date_input, date):
            return date_input
        return None

    def _get_optimization_inputs(self, analysis_date=None):
        """Obtiene Mu y Sigma filtrando correctamente por fecha."""
        logger.info(f"📥 Recuperando inputs de Black-Litterman... (Fecha: {analysis_date if analysis_date else 'HOY'})")
        
        # 1. Obtener Retornos Posteriores (Mu) desde BL
        df_bl_results = self.bl_model.run_optimization(analysis_date)
        
        if df_bl_results.is_empty():
            raise ValueError("❌ Black-Litterman no generó retornos.")

        df_bl_results = df_bl_results.sort("Ticker")
        tickers = df_bl_results["Ticker"].to_list()
        mu = df_bl_results["BL_Post_%"].to_numpy() / 100.0
        
        # 2. Obtener Matriz de Covarianzas (Sigma)
        df_returns, _ = self.bl_model._get_historical_returns()
        
        # --- CORRECCIÓN ROBUSTA DE FECHAS ---
        if analysis_date:
            target_date = self._parse_date(analysis_date)
            if target_date:
                # Aseguramos que la columna sea Date y filtramos comparando objetos Date
                df_returns = df_returns.filter(
                    pl.col("Date").cast(pl.Date) <= target_date
                )
            
        # Seleccionamos solo las columnas relevantes
        df_returns = df_returns.select(tickers)
        
        # Calculamos Sigma
        sigma = self.bl_model._estimate_covariance_scaled(df_returns)
        
        return tickers, mu, sigma

    def _calculate_metrics(self, weights, mu, sigma):
        p_ret = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return p_ret, p_vol

    # --- FUNCIONES OBJETIVO ---
    def _obj_neg_sharpe(self, weights, mu, sigma, rf):
        ret, vol = self._calculate_metrics(weights, mu, sigma)
        if vol == 0: return 1e6
        return -((ret - rf) / vol)

    def _obj_neg_return(self, weights, mu, sigma):
        ret, _ = self._calculate_metrics(weights, mu, sigma)
        return -ret

    def _obj_min_volatility(self, weights, mu, sigma):
        _, vol = self._calculate_metrics(weights, mu, sigma)
        return vol

    def optimize_portfolio(self, analysis_date=None) -> pl.DataFrame:
        """Ejecuta la optimización convexa."""
        # Propagamos la fecha
        tickers, mu, sigma = self._get_optimization_inputs(analysis_date)
        n_assets = len(tickers)
        
        objective = self.params["OBJECTIVE"]
        logger.info(f"⚖️ Iniciando optimización. Objetivo: {objective} | Activos: {n_assets}")

        horizon_days = self.bl_config["OPTIMIZATION_HORIZON"]
        rf_annual = self.params["RISK_FREE_RATE_ANNUAL"]
        rf_period = rf_annual * (horizon_days / 252.0)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if self.params["ALLOW_SHORTS"]:
            bounds = tuple((self.params["MIN_WEIGHT_PER_ASSET"], self.params["MAX_WEIGHT_PER_ASSET"]) for _ in range(n_assets))
        else:
            min_w = max(0.0, self.params["MIN_WEIGHT_PER_ASSET"])
            bounds = tuple((min_w, self.params["MAX_WEIGHT_PER_ASSET"]) for _ in range(n_assets))

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
            raise ValueError(f"Objetivo desconocido: {objective}")

        opt_result = minimize(
            fun, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-8
        )

        optimal_weights = opt_result.x
        optimal_weights[np.abs(optimal_weights) < 0.0001] = 0.0
        
        final_ret, final_vol = self._calculate_metrics(optimal_weights, mu, sigma)
        final_sharpe = (final_ret - rf_period) / final_vol if final_vol > 0 else 0

        results = []
        for i, t in enumerate(tickers):
            results.append({
                "Ticker": t,
                "Weight_%": round(optimal_weights[i] * 100, 2),
                "Exp_Ret_Period_%": round(mu[i] * 100, 3)
            })
            
        df_weights = pl.DataFrame(results).sort("Weight_%", descending=True)
        
        print("\n" + "="*60)
        print(f"💎 CARTERA ÓPTIMA ({objective})")
        print(f"   Horizonte: {horizon_days} días | Sharpe: {final_sharpe:.4f}")
        print("="*60)
        print(df_weights)
        
        return df_weights

def run_portfolio_optimization():
    optimizer = PortfolioOptimizer()
    return optimizer.optimize_portfolio()

if __name__ == "__main__":
    run_portfolio_optimization()