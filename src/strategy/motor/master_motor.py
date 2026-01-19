import sys
import os
import logging
import polars as pl
from datetime import datetime, date

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
    # Importamos el Sizer que a su vez llama al Optimizer y al Return Engine
    from src.strategy.motor.sizing import PositionSizer
except ImportError as e:
    print(f"❌ Error crítico importando módulos en Master: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MASTER_ENGINE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterEngine")

class StrategyMasterEngine:
    def __init__(self):
        """
        Orquestador Principal de la Estrategia.
        1. Predicción (ML) -> 2. Optimización (Markowitz) -> 3. Sizing (Vol Target) -> 4. Ejecución (Órdenes)
        """
        self.exec_config = strat_config.EXECUTION_CONFIG
        
        # Instanciamos el motor de sizing (que encapsula toda la lógica matemática)
        self.sizer = PositionSizer()
        
        # Configurar directorio de salida de órdenes
        # Se asegura de crear la carpeta si no existe
        self.orders_path = os.path.join(strategy_dir, self.exec_config.get("ORDERS_DIR", "orders"))
        os.makedirs(self.orders_path, exist_ok=True)

    def _is_market_open(self) -> bool:
        """Verifica si hoy es fin de semana."""
        today = datetime.now()
        # 5 = Sábado, 6 = Domingo
        if today.weekday() >= 5: 
            logger.warning(f"📅 Hoy es fin de semana ({today.strftime('%A')}). Mercado Cerrado.")
            return False
        return True

    def _filter_orders(self, df_alloc: pl.DataFrame, date_str: str) -> pl.DataFrame:
        """
        Convierte la asignación de cartera en una lista de órdenes ejecutables.
        Filtra cash y posiciones despreciables.
        """
        min_val = self.exec_config["MIN_ORDER_VALUE"]
        
        # 1. Filtrar Cash
        df_orders = df_alloc.filter(pl.col("Ticker") != "CASH (USD)")
        
        # 2. Filtrar órdenes muy pequeñas (ruido)
        df_orders = df_orders.filter(pl.col("Capital_Alloc").abs() >= min_val)
        
        # 3. Añadir Metadatos de Orden
        df_orders = df_orders.with_columns([
            pl.lit(date_str).alias("Date"),
            pl.lit("MKT").alias("Order_Type") # Asumimos Market Orders por defecto
        ])
        
        return df_orders

    def run(self, analysis_date=None):
        """
        Ejecuta el pipeline completo.
        :param analysis_date: Fecha 'YYYY-MM-DD'. Si es None, usa HOY.
        """
        # Determinar fecha de trabajo
        if analysis_date:
            current_date_str = analysis_date
        else:
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            
            # Solo chequeamos mercado abierto si estamos corriendo en "tiempo real" (sin fecha forzada)
            if self.exec_config["CHECK_MARKET_OPEN"]:
                if not self._is_market_open():
                    return None

        logger.info(f"🚀 INICIANDO ESTRATEGIA BENQ-CORE PARA: {current_date_str}")
        

        try:
            # --- PASO 1, 2 & 3: SIZING (Incluye Predicción y Optimización) ---
            logger.info("⚙️ Ejecutando Pipeline Matemático (ML + Markowitz + VolTarget)...")
            
            df_portfolio = self.sizer.get_final_allocations(analysis_date=current_date_str)
            
            if df_portfolio.is_empty():
                logger.error("❌ El motor no generó resultados (Posible falta de datos o error en modelos).")
                return None

            # --- PASO 4: GENERACIÓN DE ÓRDENES ---
            df_orders = self._filter_orders(df_portfolio, current_date_str)
            
            # --- PASO 5: PERSISTENCIA (CSV) ---
            # Guardamos dos archivos:
            # A. Estado completo de la cartera (incluyendo Cash y pesos)
            path_portfolio = os.path.join(self.orders_path, f"portfolio_target_{current_date_str}.csv")
            df_portfolio.write_csv(path_portfolio)
            
            # B. Órdenes ejecutables (para el broker/execution engine)
            path_orders = os.path.join(self.orders_path, f"orders_{current_date_str}.csv")
            df_orders.write_csv(path_orders)
            
            # --- DISPLAY EN CONSOLA ---
            print("\n" + "="*80)
            print(f"📜 TARGET PORTFOLIO GENERADO ({current_date_str})")
            print("="*80)
            
            # Columnas a mostrar (Asegurando que existen en df_portfolio)
            # Nota: 'Exp_Ret_Contrib' viene del script de Sizing
            display_cols = ["Ticker", "Role", "Vol_Adj_Weight_%", "Capital_Alloc", "Exp_Ret_Contrib"]
            
            # Intersección segura de columnas para evitar errores de display
            cols_to_show = [c for c in display_cols if c in df_orders.columns]
            
            if df_orders.is_empty():
                print("⚠️ No hay órdenes activas (Solo Cash o movimientos menores al mínimo).")
                print(df_portfolio.filter(pl.col("Ticker") == "CASH (USD)"))
            else:
                print(df_orders.select(cols_to_show))
                
            print("-" * 80)
            print(f"💾 Archivos guardados en: {self.orders_path}")
            
            logger.info("✅ Proceso maestro finalizado con éxito.")
            return df_orders 

        except Exception as e:
            logger.critical(f"❌ ERROR CRÍTICO EN MASTER MOTOR: {e}", exc_info=True)
            raise e

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Instancia y ejecución
    engine = StrategyMasterEngine()
    
    # Puedes probar una fecha específica descomentando abajo:
    # engine.run(analysis_date="2025-01-15")
    
    # Ejecución por defecto (Hoy)
    engine.run()