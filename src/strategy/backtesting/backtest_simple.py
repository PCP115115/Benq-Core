import sys
import os
import logging
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    # Importación correcta del Loader
    from src.src_DD.loader import MarketLoader
    from src.strategy.motor.sizing import PositionSizer
except ImportError as e:
    print(f"❌ Error crítico de importación: {e}")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTEST] - %(message)s')
logger = logging.getLogger("Backtester")

class BacktestEngine:
    def __init__(self):
        self.bt_config = strat_config.BACKTEST_CONFIG
        self.sizer = PositionSizer() 
        
        # 1. IMPORTANTE: Instanciar Loader sin actualizar para evitar error de master.py
        self.loader = MarketLoader(actualizar_datos=False)
        
        self.cash = self.bt_config["INITIAL_CAPITAL"]
        self.holdings = {} 
        self.portfolio_history = [] 
        
        # 2. Cargar datos históricos
        self.price_history = self._load_full_price_history()

    def _load_full_price_history(self):
        """
        Carga precios de cierre usando SQL directo (método .query del loader).
        """
        logger.info("📉 Cargando histórico de precios para valoración...")
        tickers = strat_config.TICKERS_ESTRATEGIA
        start = self.bt_config["START_DATE"]
        end = self.bt_config["END_DATE"]
        
        # Buffer de seguridad para medias móviles
        start_buffer = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        
        # --- CORRECCIÓN: USAR SQL DIRECTO ---
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
            raise ValueError("❌ No se encontraron datos. Revisa las fechas en config_strategy.")

        # Convertir a Pandas para el bucle temporal
        df_pd = df.to_pandas()
        df_pd["Date"] = pd.to_datetime(df_pd["Date"])
        
        # Pivotar: Index=Date, Cols=Tickers, Vals=Close
        df_pivot = df_pd.pivot(index="Date", columns="ticker", values="Close")
        df_pivot = df_pivot.sort_index().ffill() # Forward fill moderno
        
        return df_pivot

    def _get_rebalance_dates(self):
        """Calcula las fechas exactas donde se ejecutará la estrategia."""
        start = self.bt_config["START_DATE"]
        end = self.bt_config["END_DATE"]
        freq = self.bt_config["REBALANCE_FREQ"] 
        
        all_dates = pd.date_range(start=start, end=end, freq=freq)
        valid_dates = []
        available_dates = set(self.price_history.index.normalize())
        
        for d in all_dates:
            d_norm = d.normalize()
            if d_norm in available_dates:
                valid_dates.append(d_norm)
            else:
                for i in range(1, 6):
                    prev_d = d_norm - timedelta(days=i)
                    if prev_d in available_dates:
                        valid_dates.append(prev_d)
                        break
        
        return sorted(list(set(valid_dates)))

    def _execute_rebalance(self, date):
        """Ejecuta el motor completo para una fecha simulada."""
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"🔄 Rebalanceando: {date_str}...")
        
        try:
            # INYECCIÓN DE FECHA: Evita ver el futuro
            df_alloc = self.sizer.get_final_allocations(analysis_date=date_str)
            
            if df_alloc.is_empty():
                logger.warning(f"⚠️ Sin asignación para {date_str}")
                return

            # Valoración actual (Mark-to-Market)
            try:
                current_prices = self.price_history.loc[date]
            except KeyError:
                current_prices = self.price_history.asof(date)

            portfolio_value = self.cash
            for ticker, shares in self.holdings.items():
                if ticker in current_prices and not np.isnan(current_prices[ticker]):
                    portfolio_value += shares * current_prices[ticker]
            
            # Calcular Target Holdings
            target_holdings = {}
            df_pos = df_alloc.filter(pl.col("Ticker") != "CASH (USD)")
            
            for row in df_pos.to_dicts():
                ticker = row["Ticker"]
                target_pct = row["Vol_Adj_Weight_%"] / 100.0
                target_capital = portfolio_value * target_pct
                
                if ticker in current_prices and not np.isnan(current_prices[ticker]):
                    price = current_prices[ticker]
                    if price > 0:
                        target_holdings[ticker] = target_capital / price
            
            self._process_trades(target_holdings, current_prices)
            
        except Exception as e:
            logger.error(f"⚠️ Error en rebalanceo {date_str}: {e}")

    def _process_trades(self, target_holdings, current_prices):
        """Ejecuta cambios de posición y cobra comisiones."""
        all_tickers = set(self.holdings.keys()) | set(target_holdings.keys())
        comm_pct = self.bt_config["COMMISSION_PCT"]
        slip_pct = self.bt_config["SLIPPAGE_PCT"]
        
        trades = 0
        costs = 0.0
        
        for ticker in all_tickers:
            current_shares = self.holdings.get(ticker, 0.0)
            target_shares = target_holdings.get(ticker, 0.0)
            diff = target_shares - current_shares
            
            price = current_prices.get(ticker, 0.0)
            if price == 0: continue
            
            trade_val = abs(diff * price)
            
            if trade_val > 50: # Filtro mínimo
                fric = trade_val * (comm_pct + slip_pct)
                cost_trade = diff * price
                
                self.cash -= (cost_trade + fric)
                costs += fric
                self.holdings[ticker] = target_shares
                trades += 1
                
        if trades > 0:
            logger.info(f"   💸 Trades: {trades} | Costes: ${costs:.2f} | Caja: ${self.cash:,.2f}")

    def run_backtest(self):
        """Bucle principal."""
        logger.info(f"🚀 Iniciando Backtest ({self.bt_config['START_DATE']} -> {self.bt_config['END_DATE']})")
        
        rebalance_dates = self._get_rebalance_dates()
        logger.info(f"📅 Eventos de rebalanceo: {len(rebalance_dates)}")
        
        # Filtrar precios al rango del test
        sim_prices = self.price_history.loc[self.bt_config["START_DATE"]:self.bt_config["END_DATE"]]
        
        for date in sim_prices.index:
            # Rebalancear si toca
            if any([d.normalize() == date.normalize() for d in rebalance_dates]):
                self._execute_rebalance(date)
            
            # Valoración Diaria
            val = self.cash
            prices = sim_prices.loc[date]
            for t, s in self.holdings.items():
                if t in prices and not np.isnan(prices[t]):
                    val += s * prices[t]
            
            self.portfolio_history.append({"Date": date, "Equity": val})
            
        self._generate_report()

    def _generate_report(self):
        if not self.portfolio_history: return
        
        df = pd.DataFrame(self.portfolio_history).set_index("Date")
        df["Returns"] = df["Equity"].pct_change().fillna(0)
        df["Cum_Ret"] = (1 + df["Returns"]).cumprod()
        
        # --- CÁLCULO DEL BENCHMARK (Buy & Hold Equal Weight) ---
        # 1. Filtramos los precios al rango del backtest
        start_date = df.index[0]
        end_date = df.index[-1]
        bench_prices = self.price_history.loc[start_date:end_date].copy()
        
        # 2. Normalizamos al inicio (Base 100 o Base 1.0)
        # Invertimos 1/N en cada activo el primer día
        initial_prices = bench_prices.iloc[0]
        weights = 1.0 / len(bench_prices.columns) # Equal weight
        
        # Retornos diarios de los activos
        asset_returns = bench_prices.pct_change().fillna(0)
        
        # Retorno de la cartera Benchmark (suma ponderada de retornos)
        bench_daily_ret = asset_returns.mean(axis=1) # Media simple = Equal Weight
        bench_cum_ret = (1 + bench_daily_ret).cumprod()
        
        # --- MÉTRICAS ---
        init_cap = self.bt_config["INITIAL_CAPITAL"]
        final_cap = df["Equity"].iloc[-1]
        tot_ret = (final_cap / init_cap) - 1
        sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252) if df["Returns"].std() > 0 else 0
        dd = (df["Equity"] / df["Equity"].cummax()) - 1
        max_dd = dd.min()
        
        bench_tot_ret = bench_cum_ret.iloc[-1] - 1
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL BACKTEST")
        print("="*60)
        print(f"💰 Final Estrategia: ${final_cap:,.2f} ({tot_ret*100:.2f}%)")
        print(f"📦 Final Benchmark:  ${init_cap * (1+bench_tot_ret):,.2f} ({bench_tot_ret*100:.2f}%)")
        print(f"⚡ Sharpe: {sharpe:.2f}")
        print(f"📉 Max DD: {max_dd*100:.2f}%")
        print("="*60)
        
        # --- GRÁFICA ---
        plt.figure(figsize=(10, 6))
        
        # Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df["Cum_Ret"], label="Estrategia (Benq-Core)", color="blue", linewidth=1.5)
        plt.plot(bench_cum_ret.index, bench_cum_ret, label="Benchmark (Buy & Hold)", color="gray", linestyle="--", alpha=0.7)
        
        plt.title(f"Equity Curve | Sharpe: {sharpe:.2f}")
        plt.ylabel("Crecimiento ($1 = Base)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(df.index, dd, 0, color="red", alpha=0.3)
        plt.plot(df.index, dd, color="red", linewidth=0.5)
        plt.title(f"Drawdown (Max: {max_dd*100:.1f}%)")
        plt.ylabel("Drawdown %")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img_path = os.path.join(strategy_dir, "backtesting", "backtest_results.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
        logger.info(f"📸 Gráfico guardado en: {img_path}")
        plt.show()

if __name__ == "__main__":
    BacktestEngine().run_backtest()