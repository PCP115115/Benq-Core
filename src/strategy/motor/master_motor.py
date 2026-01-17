import sys
import os
import logging
import polars as pl
from datetime import datetime

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
    from src.strategy.motor.sizing import PositionSizer
except ImportError as e:
    print(f"❌ Error crítico importando módulos: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MASTER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterEngine")

class StrategyMasterEngine:
    def __init__(self):
        self.exec_config = strat_config.EXECUTION_CONFIG
        self.sizer = PositionSizer()
        self.orders_path = os.path.join(strategy_dir, self.exec_config["ORDERS_DIR"])
        os.makedirs(self.orders_path, exist_ok=True)

    def _is_market_open(self) -> bool:
        today = datetime.now()
        if today.weekday() >= 5: # Sábado o Domingo
            logger.warning(f"📅 Hoy es fin de semana ({today.strftime('%A')}). Mercado Cerrado.")
            return False
        return True

    def _filter_orders(self, df_alloc: pl.DataFrame) -> pl.DataFrame:
        min_val = self.exec_config["MIN_ORDER_VALUE"]
        df_orders = df_alloc.filter(pl.col("Ticker") != "CASH (USD)")
        df_orders = df_orders.filter(pl.col("Capital_Alloc").abs() >= min_val)
        
        df_orders = df_orders.with_columns([
            pl.lit(datetime.now().strftime("%Y-%m-%d")).alias("Date"),
            pl.lit("MKT").alias("Order_Type")
        ])
        return df_orders

    def run(self):
        """Ejecución principal del Pipeline."""
        logger.info("🚀 INICIANDO ESTRATEGIA BENQ-CORE")
        
        if self.exec_config["CHECK_MARKET_OPEN"]:
            if not self._is_market_open():
                logger.info("⏸️ Ejecución detenida por cierre de mercado.")
                return None # Retornamos None si está cerrado

        try:
            logger.info("⚙️ Calculando Sizing Óptimo...")
            df_portfolio = self.sizer.get_final_allocations()
            
            if df_portfolio.is_empty():
                logger.error("❌ El motor de sizing no devolvió resultados.")
                return None

            df_orders = self._filter_orders(df_portfolio)
            
            # Exportar Resultados
            date_str = datetime.now().strftime("%Y-%m-%d")
            path_portfolio = os.path.join(self.orders_path, f"portfolio_status_{date_str}.csv")
            path_orders = os.path.join(self.orders_path, f"orders_{date_str}.csv")
            
            df_portfolio.write_csv(path_portfolio)
            df_orders.write_csv(path_orders)
            
            # Reporte en Consola
            print("\n" + "="*80)
            print(f"📜 ÓRDENES GENERADAS PARA {date_str}")
            print(f"   Archivos guardados en: {self.orders_path}")
            print("="*80)
            
            if df_orders.is_empty():
                print("⚠️ No hay órdenes activas.")
            else:
                display_cols = ["Ticker", "Role", "Vol_Adj_Weight_%", "Capital_Alloc", "Exp_Ret_Horizon_%"]
                print(df_orders.select(display_cols))
                
            print("-" * 80)
            logger.info("✅ Ciclo de estrategia finalizado con éxito.")
            
            return df_orders 

        except Exception as e:
            logger.critical(f"❌ ERROR CRÍTICO EN MASTER MOTOR: {e}", exc_info=True)
            raise e


# FUNCIÓN MAESTRA FINAL:
if __name__ == "__main__":
    engine = StrategyMasterEngine()
    engine.run()