import sys
import os
import logging
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date

# --- SETUP DE RUTAS (CORREGIDO) ---
# Ubicación actual: Benq-Core/src/strategy/backtesting/backtest_simple.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../backtesting
strategy_dir = os.path.dirname(current_dir)              # .../strategy
src_dir = os.path.dirname(strategy_dir)                  # .../src
project_root = os.path.dirname(src_dir)                  # .../Benq-Core

# Añadimos la raíz del proyecto al sistema para poder hacer 'import src...'
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    import src.strategy.config_strategy as strat_config
    from src.src_DD.loader import MarketLoader
    from src.strategy.motor.sizing import PositionSizer
except ImportError as e:
    print(f"❌ Error crítico de importación: {e}")
    print(f"ℹ️ Ruta intentada: {project_root}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTEST] - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtester")

class BacktestEngine:
    def __init__(self):
        self.bt_config = strat_config.BACKTEST_CONFIG
        self.sizer = PositionSizer() 
        self.loader = MarketLoader(actualizar_datos=False)
        
        self.cash = self.bt_config["INITIAL_CAPITAL"]
        self.holdings = {} 
        self.portfolio_history = [] 
        
        # Validación de integridad de datos ANTES de empezar
        self._validate_data_availability()
        
        # Cargar datos históricos para valoración
        self.price_history = self._load_full_price_history()

    def _validate_data_availability(self):
        """
        🛑 PRE-FLIGHT CHECK ROBUSTO:
        Verifica si hay suficientes datos en el pasado manejando tipos de fecha mixtos.
        """
        logger.info("🔍 Ejecutando diagnóstico de datos...")
        start_date_cfg = datetime.strptime(self.bt_config["START_DATE"], "%Y-%m-%d")
        
        tickers = strat_config.TICKERS_ESTRATEGIA
        tickers_sql = "', '".join(tickers)
        
        query = f"SELECT min(Date) as first_date FROM market WHERE ticker IN ('{tickers_sql}')"
        res = self.loader.query(query)
        
        if res.is_empty() or res["first_date"][0] is None:
            logger.critical("❌ LA BASE DE DATOS ESTÁ VACÍA O NO CONTIENE LOS TICKERS SELECCIONADOS.")
            sys.exit(1)
            
        # --- CORRECCIÓN DE TIPO ---
        raw_date = res["first_date"][0]
        
        if isinstance(raw_date, datetime):
            first_db_date = raw_date
        elif isinstance(raw_date, date):
            first_db_date = datetime.combine(raw_date, datetime.min.time())
        elif isinstance(raw_date, str):
            # Parseo seguro si viene como string
            if " " in raw_date:
                first_db_date = datetime.strptime(raw_date.split(" ")[0], "%Y-%m-%d")
            else:
                first_db_date = datetime.strptime(raw_date, "%Y-%m-%d")
        else:
            # Fallback para tipos de pandas/numpy
            first_db_date = pd.to_datetime(raw_date).to_pydatetime()

        # Validación de margen
        days_diff = (start_date_cfg - first_db_date).days
        
        logger.info(f"📅 Inicio Backtest: {start_date_cfg.date()}")
        logger.info(f"📅 Dato más antiguo en DB: {first_db_date.date()}")
        logger.info(f"📉 Margen histórico disponible: {days_diff} días")
        
        if days_diff < 60:
            logger.error("="*60)
            logger.error("⛔ ERROR FATAL: NO HAY SUFICIENTE HISTORIAL PREVIO.")
            logger.error(f"Black-Litterman necesita calcular covarianza antes de {start_date_cfg.date()}.")
            logger.error(f"Solo tienes datos desde {first_db_date.date()}.")
            logger.error(f"💡 SOLUCIÓN: Descarga datos en tu Loader desde {start_date_cfg.year - 1}-01-01.")
            logger.error("="*60)
            sys.exit(1)
        
        logger.info("✅ Chequeo de datos exitoso. Iniciando carga...")

    def _load_full_price_history(self):
        tickers = strat_config.TICKERS_ESTRATEGIA
        start = self.bt_config["START_DATE"]
        end = self.bt_config["END_DATE"]
        
        # Buffer de seguridad para medias móviles en el gráfico de valoración
        start_buffer = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        
        tickers_sql = "', '".join(tickers)
        query = f"""
            SELECT Date, ticker, Close 
            FROM market 
            WHERE ticker IN ('{tickers_sql}') 
            AND Date >= '{start_buffer}' 
            AND Date <= '{end}'
            ORDER BY Date ASC
        """
        
        df = self.loader.query(query)
        if df.is_empty():
            raise ValueError("❌ No se encontraron precios en el rango solicitado.")

        df_pd = df.to_pandas()
        df_pd["Date"] = pd.to_datetime(df_pd["Date"])
        
        df_pivot = df_pd.pivot(index="Date", columns="ticker", values="Close")
        df_pivot = df_pivot.sort_index().ffill()
        
        return df_pivot

    def _get_rebalance_dates(self):
        start = self.bt_config["START_DATE"]
        end = self.bt_config["END_DATE"]
        freq = self.bt_config["REBALANCE_FREQ"] 
        
        all_dates = pd.date_range(start=start, end=end, freq=freq)
        valid_dates = []
        # Normalizamos índice de precios para comparación rápida
        available_dates = set(self.price_history.index.normalize())
        
        for d in all_dates:
            d_norm = d.normalize()
            # Si el día teórico cae en festivo/finde, miramos hacia atrás 5 días
            found = False
            for i in range(5):
                test_date = d_norm - timedelta(days=i)
                if test_date in available_dates:
                    valid_dates.append(test_date)
                    found = True
                    break
            if not found:
                pass
        
        return sorted(list(set(valid_dates)))

    def _execute_rebalance(self, date):
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"🔄 [REBALANCEO] Calculando señales para: {date_str}...")
        
        try:
            # --- PUNTO CRÍTICO ---
            df_alloc = self.sizer.get_final_allocations(analysis_date=date_str)
            
            if df_alloc.is_empty():
                logger.warning(f"⚠️ Black-Litterman devolvió tabla VACÍA para {date_str}.")
                logger.warning("   -> Causa probable: Datos insuficientes en esa fecha o features incompletas.")
                return

            # Mark-to-Market
            if date not in self.price_history.index:
                idx = self.price_history.index.get_indexer([date], method='ffill')[0]
                current_prices = self.price_history.iloc[idx]
            else:
                current_prices = self.price_history.loc[date]

            portfolio_value = self.cash
            # Valorar cartera actual
            for ticker, shares in self.holdings.items():
                if ticker in current_prices and not pd.isna(current_prices[ticker]):
                    portfolio_value += shares * current_prices[ticker]
            
            logger.info(f"   💰 Valor Cartera Pre-Trade: ${portfolio_value:,.2f}")

            # Calcular Target
            target_holdings = {}
            df_pos = df_alloc.filter(pl.col("Ticker") != "CASH (USD)")
            
            for row in df_pos.to_dicts():
                ticker = row["Ticker"]
                target_pct = row["Vol_Adj_Weight_%"] / 100.0
                target_capital = portfolio_value * target_pct
                
                if ticker in current_prices and not pd.isna(current_prices[ticker]):
                    price = current_prices[ticker]
                    if price > 0:
                        # Usamos int() para simular lotes enteros
                        target_holdings[ticker] = int(target_capital / price)
            
            self._process_trades(target_holdings, current_prices)
            
        except Exception as e:
            logger.error(f"❌ EXCEPCIÓN EN MOTOR DE REBALANCEO: {e}", exc_info=True)

    def _process_trades(self, target_holdings, current_prices):
        comm_pct = self.bt_config["COMMISSION_PCT"]
        slip_pct = self.bt_config["SLIPPAGE_PCT"]
        
        all_tickers = set(self.holdings.keys()) | set(target_holdings.keys())
        trades = 0
        costs = 0.0
        
        for ticker in all_tickers:
            current_shares = self.holdings.get(ticker, 0.0)
            target_shares = target_holdings.get(ticker, 0.0)
            diff = target_shares - current_shares
            
            if diff == 0: continue

            price = current_prices.get(ticker, 0.0)
            if price == 0: continue
            
            trade_val = abs(diff * price)
            
            # Filtro de ruido
            if trade_val > 50:
                fric = trade_val * (comm_pct + slip_pct)
                cost_trade = diff * price
                
                self.cash -= (cost_trade + fric)
                costs += fric
                self.holdings[ticker] = target_shares
                trades += 1
                
        if trades > 0:
            logger.info(f"   💸 Ejecutados {trades} trades | Comisiones: ${costs:.2f} | Caja Final: ${self.cash:,.2f}")

    def run_backtest(self):
        logger.info(f"🚀 INICIANDO SIMULACIÓN ({self.bt_config['START_DATE']} -> {self.bt_config['END_DATE']})")
        
        rebalance_dates = self._get_rebalance_dates()
        logger.info(f"📅 Fechas de Rebalanceo Activas: {len(rebalance_dates)}")
        if len(rebalance_dates) > 0:
            logger.info(f"   -> Primera fecha: {rebalance_dates[0].date()}")
            logger.info(f"   -> Última fecha:  {rebalance_dates[-1].date()}")
        else:
            logger.warning("⚠️ NO SE GENERARON FECHAS DE REBALANCEO. Revisa REBALANCE_FREQ.")
        
        sim_prices = self.price_history.loc[self.bt_config["START_DATE"]:self.bt_config["END_DATE"]]
        
        if sim_prices.empty:
            logger.error("❌ El rango de fechas seleccionado no tiene precios en la BD cargada.")
            return

        for date in sim_prices.index:
            if date in rebalance_dates:
                self._execute_rebalance(date)
            
            val = self.cash
            daily_prices = sim_prices.loc[date]
            for t, s in self.holdings.items():
                if t in daily_prices and not pd.isna(daily_prices[t]):
                    val += s * daily_prices[t]
            
            self.portfolio_history.append({"Date": date, "Equity": val})
            
        self._generate_report()

    def _generate_report(self):
        if not self.portfolio_history: 
            logger.error("❌ No se generó historial de cartera.")
            return
        
        df = pd.DataFrame(self.portfolio_history).set_index("Date")
        df["Returns"] = df["Equity"].pct_change().fillna(0)
        df["Cum_Ret"] = (1 + df["Returns"]).cumprod()
        
        start_date = df.index[0]
        end_date = df.index[-1]
        bench_prices = self.price_history.loc[start_date:end_date]
        bench_ret = bench_prices.pct_change().fillna(0).mean(axis=1) # Equal Weight
        bench_cum = (1 + bench_ret).cumprod()
        
        final_equity = df["Equity"].iloc[-1]
        total_ret = (final_equity / self.bt_config["INITIAL_CAPITAL"]) - 1
        bench_total_ret = bench_cum.iloc[-1] - 1
        
        sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252) if df["Returns"].std() > 0 else 0
        dd = (df["Equity"] / df["Equity"].cummax()) - 1
        max_dd = dd.min()
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL BACKTEST")
        print("="*60)
        print(f"💰 Rentabilidad Estrategia: {total_ret*100:.2f}% (${final_equity:,.0f})")
        print(f"📦 Rentabilidad Benchmark:  {bench_total_ret*100:.2f}%")
        print(f"⚡ Sharpe Ratio: {sharpe:.2f}")
        print(f"📉 Max Drawdown: {max_dd*100:.2f}%")
        print("="*60)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Cum_Ret"], label="Benq-Core Strategy", linewidth=1.5)
        plt.plot(bench_cum.index, bench_cum, label="Benchmark (EW)", color="gray", alpha=0.6, linestyle="--")
        plt.title("Curva de Equidad vs Benchmark")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    BacktestEngine().run_backtest()