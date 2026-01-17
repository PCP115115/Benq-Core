import sys
import os
import logging
import numpy as np
import polars as pl

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(current_dir)     # src/strategy
src_dir = os.path.dirname(strategy_dir)         # src
project_root = os.path.dirname(src_dir)         # Benq-Core

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    import src.strategy.config_strategy as strat_config
    # Importamos el Optimizador para obtener pesos y matriz de covarianza
    from src.strategy.motor.cartera import PortfolioOptimizer
except ImportError as e:
    print(f"❌ Error crítico de importación en Sizing: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SIZING] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VolTarget")

class PositionSizer:
    def __init__(self):
        """
        Motor de Sizing basado en Volatility Targeting.
        Escala la cartera óptima de Markowitz para cumplir un objetivo de volatilidad anual.
        """
        self.size_config = strat_config.SIZING_CONFIG
        self.bl_config = strat_config.BLACK_LITTERMAN_CONFIG
        
        # Instanciamos el optimizador (que a su vez llama a BL)
        self.optimizer = PortfolioOptimizer()

    def calculate_volatility_scalar(self, weights: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calcula el factor de escala (Leverage) necesario para alcanzar el Vol Target.
        
        Formula: Scalar = Target_Vol / Portfolio_Vol
        """
        # 1. Calcular volatilidad esperada del portafolio (en el horizonte del modelo, ej. 5 días)
        # Var = w.T * Sigma * w
        port_variance_period = np.dot(weights.T, np.dot(sigma, weights))
        port_vol_period = np.sqrt(port_variance_period)
        
        if port_vol_period == 0:
            return 0.0

        # 2. Anualizar la volatilidad del portafolio para compararla con el Target anual
        # Vol_Anual = Vol_Periodo * sqrt(252 / Dias_Horizonte)
        horizon_days = self.bl_config["OPTIMIZATION_HORIZON"]
        annualization_factor = np.sqrt(252.0 / horizon_days)
        port_vol_annual = port_vol_period * annualization_factor
        
        logger.info(f"📊 Volatilidad Cartera (Base): {port_vol_period*100:.2f}% ({horizon_days}d) -> {port_vol_annual*100:.2f}% (Anual)")

        # 3. Calcular Scalar (Multiplicador de exposición)
        target_vol = self.size_config["TARGET_VOLATILITY_ANNUAL"]
        
        # Si la vol es 0 (cartera vacía), scalar es 0
        if port_vol_annual == 0: return 0.0
        
        scalar = target_vol / port_vol_annual
        
        logger.info(f"🎯 Target Vol: {target_vol*100:.1f}% | Scalar Calculado: {scalar:.4f}x")
        
        return scalar

    def get_final_allocations(self, analysis_date=None) -> pl.DataFrame:
        """
        Genera la tabla final de asignación de capital.
        """
        # 1. Obtener la Cartera Óptima (Pesos Relativos) y Datos Matemáticos
        # optimize_portfolio devuelve DF, pero necesitamos Sigma y Mu crudos para recalcular riesgo
        # Así que llamamos a los métodos internos del optimizador para precisión.
        
        logger.info("⚖️ Obteniendo cartera óptima base (Markowitz)...")
        
        tickers, mu, sigma = self.optimizer._get_optimization_inputs(analysis_date)
        df_optimal = self.optimizer.optimize_portfolio(analysis_date)
        
        # Reordenamos df_optimal para que coincida con el orden de 'tickers' y 'sigma'
        # Esto es crítico para que el producto matricial sea correcto
        ticker_weight_map = {row["Ticker"]: row["Weight_%"] for row in df_optimal.to_dicts()}
        weights_vector = np.array([ticker_weight_map.get(t, 0.0) / 100.0 for t in tickers])
        
        # 2. Calcular Scalar de Volatilidad
        scalar = self.calculate_volatility_scalar(weights_vector, sigma)
        
        # 3. Aplicar Restricciones de Apalancamiento (Capping)
        max_lev = self.size_config["MAX_LEVERAGE"]
        
        # Si tenemos un buffer de cash obligatorio (ej. 2%), el leverage max efectivo se reduce ligeramente
        # para garantizar liquidez operativa.
        effective_cap = max_lev * (1.0 - self.size_config["MIN_CASH_BUFFER"])
        
        final_leverage = min(scalar, effective_cap)
        
        logger.info(f"🔒 Leverage Final Aplicado: {final_leverage:.4f}x (Max Config: {max_lev}x)")

        # 4. Calcular Allocations Finales
        total_capital = self.size_config["TOTAL_CAPITAL"]
        
        results = []
        exposure_pct_total = 0.0
        
        for i, ticker in enumerate(tickers):
            base_weight = weights_vector[i]
            
            # Peso final = Peso Relativo * Factor de Apalancamiento
            # Ej: Si Markowitz dice 40% y Leverage es 1.5 -> Peso Final 60%
            final_weight = base_weight * final_leverage
            
            # Dinero a invertir
            cash_allocation = final_weight * total_capital
            
            if abs(final_weight) > 0.0001: # Filtrar posiciones nulas
                results.append({
                    "Ticker": ticker,
                    "Role": "LONG" if final_weight > 0 else "SHORT",
                    "Base_Weight_%": round(base_weight * 100, 2),
                    "Vol_Adj_Weight_%": round(final_weight * 100, 2),
                    "Capital_Alloc": round(cash_allocation, 2),
                    # View esperada (solo informativo, no afecta al sizing aquí)
                    "Exp_Ret_Horizon_%": round(mu[i] * 100, 3) 
                })
                exposure_pct_total += abs(final_weight)

        # Añadir fila de CASH (Residual)
        # Cash = 100% - Exposición Total (puede ser negativo si usamos margen > 1.0)
        cash_residual_pct = 1.0 - exposure_pct_total
        cash_residual_val = total_capital * cash_residual_pct
        
        results.append({
            "Ticker": "CASH (USD)",
            "Role": "LIQUIDITY",
            "Base_Weight_%": 0.0,
            "Vol_Adj_Weight_%": round(cash_residual_pct * 100, 2),
            "Capital_Alloc": round(cash_residual_val, 2),
            "Exp_Ret_Horizon_%": 0.0
        })

        df_sizing = pl.DataFrame(results).sort("Capital_Alloc", descending=True)
        
        print("\n" + "="*80)
        print(f"💰 SIZING FINAL (VOL TARGET {self.size_config['TARGET_VOLATILITY_ANNUAL']*100}%)")
        print(f"   Capital Total: ${total_capital:,.2f}")
        print(f"   Exposición Total: {exposure_pct_total*100:.2f}% | Leverage: {final_leverage:.3f}x")
        print("="*80)
        print(df_sizing)
        
        return df_sizing

# --- ENTRY POINT ---
def get_position_sizes():
    sizer = PositionSizer()
    return sizer.get_final_allocations()

if __name__ == "__main__":
    get_position_sizes()