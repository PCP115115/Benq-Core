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
    # Importamos la clase PortfolioOptimizer actualizada
    from src.strategy.motor.cartera import PortfolioOptimizer
except ImportError as e:
    print(f"❌ Error crítico de importación en Sizing: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SIZING_ENGINE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VolTarget")

class PositionSizer:
    def __init__(self):
        """
        Motor de Sizing: Ajusta la exposición total de la cartera (Leverage)
        para cumplir con un objetivo de volatilidad anual (Volatility Targeting).
        """
        self.size_config = strat_config.SIZING_CONFIG
        # Necesitamos el horizonte de optimización definido en BL Config
        self.bl_config = strat_config.BLACK_LITTERMAN_CONFIG 
        
        # Instanciamos el optimizador (Markowitz)
        self.optimizer = PortfolioOptimizer()

    def calculate_volatility_scalar(self, weights: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calcula el factor de escala (Leverage) necesario para igualar la Target Vol.
        """
        # 1. Calcular Varianza del Periodo (Horizonte 5 días)
        # Var = w^T * Sigma * w
        port_variance_period = np.dot(weights.T, np.dot(sigma, weights))
        port_vol_period = np.sqrt(port_variance_period)
        
        if port_vol_period == 0:
            return 0.0

        # 2. Anualizar la Volatilidad
        # Sigma viene escalada al horizonte (ej. 5 días).
        # Factor = sqrt(252 / 5)
        horizon_days = self.bl_config.get("OPTIMIZATION_HORIZON", 5)
        annualization_factor = np.sqrt(252.0 / horizon_days)
        
        port_vol_annual = port_vol_period * annualization_factor
        
        # 3. Calcular Scalar (Leverage Ratio)
        target_vol = self.size_config["TARGET_VOLATILITY_ANNUAL"]
        
        if port_vol_annual == 0: 
            return 0.0
        
        # Si la cartera es muy tranquila (vol 5%), y target es 20%, scalar será 4.0x
        # Si la cartera es muy volátil (vol 40%), y target es 20%, scalar será 0.5x
        scalar = target_vol / port_vol_annual
        
        # logger.info(f"📊 Vol Cartera (Anual): {port_vol_annual:.2%} | Target: {target_vol:.2%} | Scalar Bruto: {scalar:.2f}x")
        return scalar

    def get_final_allocations(self, analysis_date=None) -> pl.DataFrame:
        """
        Orquesta el proceso: Markowitz -> Vol Targeting -> Cap de Apalancamiento -> Cash Management.
        """
        date_str = analysis_date if analysis_date else "HOY"
        # logger.info(f"⚖️ Iniciando cálculo de Sizing para {date_str}...")
        
        # 1. Obtener Matriz de Riesgo (Sigma)
        # Llamamos al método interno del optimizador para obtener la Sigma actual (con Shrinkage)
        tickers, mu, sigma = self.optimizer._get_optimization_inputs(analysis_date)
        
        if tickers is None or sigma is None:
            logger.warning("⚠️ No se pudieron obtener inputs de riesgo. Abortando Sizing.")
            return pl.DataFrame()

        # 2. Obtener Pesos Óptimos (Forma de la Cartera)
        # Esto ejecuta Markowitz (maximize Sharpe)
        df_optimal = self.optimizer.optimize(analysis_date)
        
        if df_optimal.is_empty():
            logger.warning("⚠️ Optimización devolvió cartera vacía.")
            return pl.DataFrame()
        
        # 3. Alinear Pesos con Tickers
        # df_optimal puede tener menos filas si filtra pesos cero, o un orden distinto.
        # Creamos un vector de pesos alineado con la lista 'tickers' y la matriz 'sigma'
        
        # Usamos columna "Weight" (decimal)
        weight_map = {row["Ticker"]: row["Weight"] for row in df_optimal.to_dicts()}
        weights_vector = np.array([weight_map.get(t, 0.0) for t in tickers])
        
        # 4. Calcular Apalancamiento Objetivo (Vol Target)
        scalar = self.calculate_volatility_scalar(weights_vector, sigma)
        
        # 5. Aplicar Límites de Apalancamiento (Safety Caps)
        max_lev = self.size_config["MAX_LEVERAGE"]
        min_cash_buffer = self.size_config["MIN_CASH_BUFFER"]
        
        # El tope efectivo debe dejar espacio para el cash buffer si vamos al máximo
        # Ej: Max Lev 1.5, pero queremos 2% cash -> Max real = 1.5 * 0.98? 
        # Simplificación: Limitamos el leverage total y luego aseguramos cash si es Long Only.
        
        final_leverage = min(scalar, max_lev)
        
        # logger.info(f"🔒 Leverage Aplicado: {final_leverage:.2f}x (Max Config: {max_lev}x)")

        # 6. Construir Allocations Finales
        total_capital = self.size_config["TOTAL_CAPITAL"]
        results = []
        gross_exposure_pct = 0.0
        
        for i, ticker in enumerate(tickers):
            base_weight = weights_vector[i]
            
            # Peso Final = Peso Optimo * Leverage
            final_weight_pct = base_weight * final_leverage
            
            # Dinero = % Final * Capital Total
            cash_allocation = final_weight_pct * total_capital
            
            if abs(final_weight_pct) > 0.0001:
                results.append({
                    "Ticker": ticker,
                    "Role": "LONG" if final_weight_pct > 0 else "SHORT",
                    "Base_Weight_%": round(base_weight * 100, 2),
                    "Vol_Adj_Weight_%": round(final_weight_pct * 100, 2),
                    "Capital_Alloc": round(cash_allocation, 2),
                    # Retorno esperado escalado al peso final (Opcional, informativo)
                    "Exp_Ret_Contrib": round(mu[i] * final_weight_pct * 100, 4) 
                })
                gross_exposure_pct += abs(final_weight_pct)

        # 7. Gestión de Liquidez (Cash)
        # El cash es lo que sobra: 100% - Exposición Neta (si es Long Only)
        # Si hay cortos, el cálculo de cash es distinto (margin), 
        # pero para una cuenta simple: Equity - Market Value of Positions.
        
        # Calculamos el Cash Teórico "Libre"
        # Si leverage > 1, estamos tomando prestado, el cash "contable" baja o se vuelve negativo (margin loan).
        market_value_total = sum([r["Capital_Alloc"] for r in results])
        cash_residual_val = total_capital - market_value_total
        
        results.append({
            "Ticker": "CASH (USD)",
            "Role": "LIQUIDITY",
            "Base_Weight_%": 0.0,
            "Vol_Adj_Weight_%": round((cash_residual_val / total_capital) * 100, 2),
            "Capital_Alloc": round(cash_residual_val, 2),
            "Exp_Ret_Contrib": 0.0
        })

        df_sizing = pl.DataFrame(results).sort("Capital_Alloc", descending=True)
        return df_sizing

# --- ENTRY POINT ---
def get_position_sizes(analysis_date=None):
    """Función maestra para obtener el dimensionamiento final."""
    sizer = PositionSizer()
    return sizer.get_final_allocations(analysis_date=analysis_date)

if __name__ == "__main__":
    # Test manual
    print("🚀 Calculando Sizing con Volatilidad Objetivo...")
    df = get_position_sizes()
    print(df)