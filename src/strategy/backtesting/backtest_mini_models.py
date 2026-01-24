# src/strategy/backtesting/backtest_mini_models.py

import sys
import os
import logging
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# --- 1. SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
strategy_dir = os.path.dirname(current_dir)              
src_dir = os.path.dirname(strategy_dir)                  
project_root = os.path.dirname(src_dir)                  

if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. IMPORTS DEL PROYECTO ---
try:
    import src.strategy.config_strategy as strat_config
    import src.engine.config as engine_config
    from src.src_DD.loader import MarketLoader
    from src.engine.src_features import master_features, indicators
except ImportError as e:
    print(f"❌ Error crítico de importación: {e}")
    sys.exit(1)

# --- 3. LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WF_BACKTEST] - %(levelname)s - %(message)s')
logger = logging.getLogger("WalkForwardBacktest")

class MiniModelWalkForward:
    def __init__(self):
        print("\n🔍 VERSIÓN DEBUG ACTIVADA: Si no ves esto, no estás ejecutando el archivo correcto.\n")
        self.tickers = strat_config.TICKERS_ESTRATEGIA
        self.bt_config = strat_config.BACKTEST_CONFIG
        self.pf_config = strat_config.PORTFOLIO_CONFIG
        self.size_config = strat_config.SIZING_CONFIG
        self.mini_config = engine_config.MINI_MODEL_PARAMS
        self.feat_params = engine_config.FEATURES_PARAMS
        
        self.loader = MarketLoader(actualizar_datos=False)
        self.full_feature_matrices = {} 
        self.price_history = None
        
        self._load_data()

    def _load_data(self):
        logger.info("⏳ Cargando histórico de precios y calculando features maestras...")
        start_date = self.bt_config["START_DATE"]
        end_date = self.bt_config["END_DATE"]
        start_buffer = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        
        tickers_sql = "', '".join(self.tickers)
        query = f"""
            SELECT Date, ticker, Close 
            FROM market 
            WHERE ticker IN ('{tickers_sql}') 
            AND Date >= '{start_buffer}' 
            AND Date <= '{end_date}'
            ORDER BY Date ASC
        """
        df_prices = self.loader.query(query)
        if df_prices is None or df_prices.is_empty():
            logger.critical("❌ NO SE HAN CARGADO PRECIOS DE LA BASE DE DATOS.")
            sys.exit(1)

        self.price_history = df_prices.to_pandas().pivot(index="Date", columns="ticker", values="Close").ffill()
        self.price_history.index = pd.to_datetime(self.price_history.index)

        layer = self.mini_config["LAYER"][0]
        
        for ticker in self.tickers:
            logger.info(f"   -> Generando matriz de features para {ticker}...")
            df_feat = master_features.get_feature_matrix(tickers=ticker, layer=layer, features=None)
            
            if df_feat is not None:
                vol_col = f"vol_yz_{self.feat_params['YANG_ZHANG_WINDOW']}d"
                horizon = self.mini_config["FORECAST_HORIZON"]
                z_score = self.feat_params["YZ_Z_SCORE"]
                
                try:
                    bounds_exprs = indicators.get_volatility_bounds(
                        col_close="Close", col_vol_yz=vol_col, z_score=z_score, horizon=horizon
                    )
                    df_feat = df_feat.with_columns(bounds_exprs)
                    df_feat = df_feat.with_columns(pl.col("Date").cast(pl.Datetime))
                    self.full_feature_matrices[ticker] = df_feat
                except Exception as e:
                    logger.error(f"⚠️ Error calculando features extra para {ticker}: {e}")
            else:
                logger.warning(f"⚠️ No se pudieron generar features para {ticker}")

    def _generate_targets(self, df: pl.DataFrame, horizon: int):
        ceil_col = f"fprice_ceil_yz_{horizon}d"
        floor_col = f"fprice_floor_yz_{horizon}d"
        
        days_to_ceil = []
        days_to_floor = []
        
        for i in range(1, horizon + 1):
            future_high = pl.col("High").shift(-i)
            future_low = pl.col("Low").shift(-i)
            
            hit_c = pl.when(future_high > pl.col(ceil_col)).then(i).otherwise(999)
            hit_f = pl.when(future_low < pl.col(floor_col)).then(i).otherwise(999)
            
            days_to_ceil.append(hit_c)
            days_to_floor.append(hit_f)
            
        df = df.with_columns([
            pl.min_horizontal(days_to_ceil).alias("first_ceil_hit"),
            pl.min_horizontal(days_to_floor).alias("first_floor_hit")
        ])
        
        target_up = ((pl.col("first_ceil_hit") != 999) & (pl.col("first_ceil_hit") < pl.col("first_floor_hit"))).cast(pl.Int8)
        target_down = ((pl.col("first_floor_hit") != 999) & (pl.col("first_floor_hit") < pl.col("first_ceil_hit"))).cast(pl.Int8)
        
        return df.with_columns([
            target_up.alias("target_up"),
            target_down.alias("target_down")
        ])

    def _train_and_predict(self, ticker, current_date, strategy_type="TREND"):
        # --- VERSIÓN DEBUG FORZADA ---
        df_full = self.full_feature_matrices.get(ticker)
        if df_full is None: 
            # print(f"🔴 [{ticker}] No hay DataFrame de features.")
            return 0.0, 0.0, 0.0, 0.0

        horizon = self.mini_config["FORECAST_HORIZON"]
        
        # Filtramos estrictamente el pasado
        df_hist = df_full.filter(pl.col("Date") <= current_date)
        
        # CHEQUEO 1: Datos insuficientes
        if df_hist.height < 252: 
            if df_hist.height > 0: # Solo avisar si hay ALGO de datos pero no suficientes
                print(f"🟠 [{ticker}] Historial insuficiente ({df_hist.height} filas) para fecha {current_date.date()}")
            return 0.0, 0.0, 0.0, 0.0 
        
        # Generar Targets
        df_train_w_targets = self._generate_targets(df_hist, horizon)
        
        train_cutoff_idx = df_train_w_targets.height - horizon
        df_train_final = df_train_w_targets.slice(0, train_cutoff_idx)
        
        if strategy_type == "TREND":
            features_cols = self.mini_config["FEATURES_TREND"]
        else: 
            features_cols = self.mini_config["FEATURES_REVERSION"]
            
        try:
            # CHEQUEO 2: Columnas faltantes (Esto suele ser el error silencioso)
            missing = [c for c in features_cols if c not in df_train_final.columns]
            if missing:
                print(f"🔴 [{ticker}] FALTAN COLUMNAS DE FEATURES: {missing}")
                return 0.0, 0.0, 0.0, 0.0

            df_train_final = df_train_final.drop_nulls(subset=features_cols + ["target_up", "target_down"])
        except Exception as e:
            print(f"🔴 [{ticker}] Error limpiando nulos: {e}")
            return 0.0, 0.0, 0.0, 0.0
        
        if df_train_final.height < 100: 
            print(f"🟠 [{ticker}] Dataset vacío tras limpieza (Posible falta de indicadores técnicos).")
            return 0.0, 0.0, 0.0, 0.0

        try:
            # Entrenamiento rápido
            X_train = df_train_final.select(features_cols).to_pandas()
            y_up = df_train_final["target_up"].to_pandas()
            y_down = df_train_final["target_down"].to_pandas()
            
            params = self.mini_config["LGBM_PARAMS"]
            params["verbosity"] = -1
            
            model_up = lgb.LGBMClassifier(**params)
            model_up.fit(X_train, y_up)
            
            model_down = lgb.LGBMClassifier(**params)
            model_down.fit(X_train, y_down)
        except Exception as e:
            print(f"🔴 [{ticker}] Error entrenando LGBM: {e}")
            return 0.0, 0.0, 0.0, 0.0
        
        row_current = df_hist.filter(pl.col("Date") == current_date)
        
        if row_current.height == 0: 
            # print(f"🟠 [{ticker}] No hay datos para HOY ({current_date.date()})")
            return 0.0, 0.0, 0.0, 0.0
        
        X_curr = row_current.select(features_cols).to_pandas()
        
        p_up = model_up.predict_proba(X_curr)[0][1]
        p_down = model_down.predict_proba(X_curr)[0][1]
        
        ceil_col = f"fprice_ceil_yz_{horizon}d"
        floor_col = f"fprice_floor_yz_{horizon}d"
        
        price = row_current["Close"][0]
        ceil = row_current[ceil_col][0]
        floor = row_current[floor_col][0]
        
        if price <= 0: return 0.0, 0.0, 0.0, 0.0
        
        r_tp = (ceil - price) / price  
        r_dw = (price - floor) / price 
        r_dw_mag = abs((price - floor) / price)
        
        return p_up, p_down, r_tp, r_dw_mag

    def _optimize_portfolio(self, mu_dict, cov_matrix, current_date, current_capital):
        tickers = list(mu_dict.keys())
        n_assets = len(tickers)
        if n_assets == 0: return {}
        
        mu_vec = np.array([mu_dict[t] for t in tickers])

        lookback = 252
        start_hist = current_date - timedelta(days=lookback + 20)
        df_returns_list = []
        for t in tickers:
            if t in self.full_feature_matrices:
                df_t = self.full_feature_matrices[t]
                df_slice = df_t.filter((pl.col("Date") >= start_hist) & (pl.col("Date") < current_date)).select("log_returns")
                if df_slice.height > 60: df_returns_list.append(df_slice.to_numpy().flatten())
                else: return {}
        
        if not df_returns_list: return {}
        
        min_len = min([len(x) for x in df_returns_list])
        X = np.column_stack([x[-min_len:] for x in df_returns_list])
        
        try:
            lw = LedoitWolf()
            sigma_daily = lw.fit(X).covariance_
            horizon = self.mini_config["FORECAST_HORIZON"]
            sigma_period = sigma_daily * horizon
        except:
            horizon = self.mini_config["FORECAST_HORIZON"]
            sigma_period = np.cov(X, rowvar=False) * horizon

        rf = self.pf_config["RISK_FREE_RATE_ANNUAL"] * (horizon / 365.0)
        
        def neg_sharpe(w):
            p_ret = np.sum(mu_vec * w)
            p_vol = np.sqrt(np.dot(w.T, np.dot(sigma_period, w)))
            if p_vol < 1e-6: return 1e6
            return -((p_ret - rf) / p_vol)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(n_assets)) 
        
        init_guess = np.array([1/n_assets]*n_assets)
        try:
            res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
            opt_weights = res.x
        except:
            return {}
        
        port_var = np.dot(opt_weights.T, np.dot(sigma_period, opt_weights))
        port_vol_annual = np.sqrt(port_var) * np.sqrt(252/horizon)
        
        target_vol = self.size_config["TARGET_VOLATILITY_ANNUAL"]
        max_lev = self.size_config["MAX_LEVERAGE"]
        
        scalar = target_vol / port_vol_annual if port_vol_annual > 0 else 0
        final_lev = min(scalar, max_lev)
        
        allocations = {}
        for i, t in enumerate(tickers):
            weight = opt_weights[i] * final_lev
            allocations[t] = weight * current_capital
            
        invested = sum(allocations.values())
        allocations["CASH"] = current_capital - invested
        
        return allocations

    def run_strategy(self, strategy_name="TREND"):
        logger.info(f"🚀 INICIANDO BACKTEST: ESTRATEGIA {strategy_name}")
        
        rebal_freq = self.bt_config["REBALANCE_FREQ"]
        dates = pd.date_range(start=self.bt_config["START_DATE"], end=self.bt_config["END_DATE"], freq=rebal_freq)
        
        history = []
        
        current_cash = self.bt_config["INITIAL_CAPITAL"]
        current_shares = {} 
        
        for d in dates:
            d_pl = datetime(d.year, d.month, d.day)
            
            # --- 1. OBTENER PRECIOS DEL MERCADO (CORREGIDO) ---
            if d in self.price_history.index:
                row_prices = self.price_history.loc[d]
            else:
                idx = self.price_history.index.get_indexer([d], method='pad')[0]
                row_prices = self.price_history.iloc[idx]
            
            # ¡FIX CRÍTICO!: Convertimos TODOS los precios disponibles a diccionario
            # Ahora el modelo ve los precios de todo, no solo de lo que tiene en cartera.
            current_prices = row_prices.dropna().to_dict()

            # --- 2. Mark-to-Market (Valorar Cartera) ---
            portfolio_value = current_cash
            for ticker, shares in current_shares.items():
                price = current_prices.get(ticker, 0.0)
                portfolio_value += shares * price
            
            history.append({"Date": d, "Equity": portfolio_value})
            
            # --- 3. Predicciones y Señales ---
            mu_dict = {}
            for ticker in self.tickers:
                try:
                    # Filtro de seguridad: Si no hay precio hoy, no podemos operar
                    if ticker not in current_prices: continue 

                    p_up, p_down, r_tp, r_dw = self._train_and_predict(ticker, d_pl, strategy_name)
                    
                    # Fórmula de Retorno Esperado
                    expected_return = (p_up * r_tp) - (p_down * r_dw)
                    
                    # --- 🕵️ CÓDIGO ESPÍA (Opcional: Descomentar si quieres ver los cálculos) ---
                    # if ticker == "AAPL":
                    #     print(f"   [AAPL] {d.date()} | E[R]: {expected_return:.4f} (Pup:{p_up:.2f} Pdw:{p_down:.2f})")
                    
                    # Umbral de entrada (ligeramente positivo para filtrar ruido)
                    if expected_return > 0.0005: 
                        mu_dict[ticker] = expected_return
                        
                except Exception as e:
                    print(f"ERROR FATAL en bucle {ticker}: {e}")
            
            # --- 4. Optimización de Cartera ---
            target_allocations = {}
            if mu_dict:
                # print(f"   ✅ Señales: {len(mu_dict)} activos seleccionados.")
                target_allocations = self._optimize_portfolio(mu_dict, None, d_pl, portfolio_value)
            
            # --- 5. Ejecución (Rebalanceo) ---
            if not target_allocations:
                # Si no hay señales, nos vamos a CASH
                current_cash = portfolio_value
                current_shares = {}
            else:
                new_shares = {}
                new_cash = target_allocations.get("CASH", 0.0)
                
                for ticker, alloc_amt in target_allocations.items():
                    if ticker == "CASH": continue
                    price = current_prices.get(ticker, 0.0)
                    if price > 0:
                        new_shares[ticker] = alloc_amt / price
                    else:
                        # Si falla el precio, devolvemos al cash
                        new_cash += alloc_amt
                
                current_shares = new_shares
                current_cash = new_cash

        return pd.DataFrame(history).set_index("Date")
    
    def _calculate_metrics(self, df, initial_cap):
        if df.empty: return 0.0, 0.0, 0.0
        
        total_ret = (df["Equity"].iloc[-1] / initial_cap) - 1
        pct_change = df["Equity"].pct_change().fillna(0)
        
        freq_factor = 12 if self.bt_config["REBALANCE_FREQ"] == "MS" else 52
        std_dev = pct_change.std()
        sharpe = (pct_change.mean() / std_dev) * np.sqrt(freq_factor) if std_dev > 0 else 0
        
        cum_max = df["Equity"].cummax()
        drawdown = (df["Equity"] / cum_max) - 1
        max_dd = drawdown.min()
        
        return total_ret, sharpe, max_dd
    

    def run_comparison(self):
        df_trend = self.run_strategy("TREND")
        df_rev = self.run_strategy("REVERSION")
        initial_cap = self.bt_config["INITIAL_CAPITAL"]
        
        logger.info("📊 Calculando Benchmark (Equal Weight)...")
        bench_hist = []
        start_date = pd.to_datetime(self.bt_config["START_DATE"])
        
        prices_start = self.price_history.loc[self.price_history.index >= start_date].iloc[0]
        shares = (initial_cap / len(self.tickers)) / prices_start
        
        ref_index = df_trend.index if not df_trend.empty else self.price_history.loc[start_date:].index
        
        for d in ref_index:
            if d in self.price_history.index:
                prices = self.price_history.loc[d]
            else:
                idx = self.price_history.index.get_indexer([d], method='pad')[0]
                prices = self.price_history.iloc[idx]
            
            val = (prices * shares).sum()
            bench_hist.append({"Date": d, "Equity": val})

        df_bench = pd.DataFrame(bench_hist).set_index("Date")
        
        ret_trend, sharpe_trend, dd_trend = self._calculate_metrics(df_trend, initial_cap)
        ret_rev, sharpe_rev, dd_rev = self._calculate_metrics(df_rev, initial_cap)
        ret_bench, sharpe_bench, dd_bench = self._calculate_metrics(df_bench, initial_cap)
        
        print("\n" + "="*60)
        print("📊 RESULTADOS FINAL DEL WALK-FORWARD (MINI-MODELS)")
        print("="*60)
        
        print(f"🔹 ESTRATEGIA TREND:")
        print(f"   💰 Retorno Total: {ret_trend:.2%} (${df_trend['Equity'].iloc[-1]:,.0f})")
        print(f"   ⚡ Sharpe Ratio:  {sharpe_trend:.2f}")
        print(f"   📉 Max Drawdown:  {dd_trend:.2%}")
        print("-" * 30)
        
        print(f"🔸 ESTRATEGIA REVERSION:")
        print(f"   💰 Retorno Total: {ret_rev:.2%} (${df_rev['Equity'].iloc[-1]:,.0f})")
        print(f"   ⚡ Sharpe Ratio:  {sharpe_rev:.2f}")
        print(f"   📉 Max Drawdown:  {dd_rev:.2%}")
        print("-" * 30)
        
        print(f"📦 BENCHMARK (EW Buy & Hold):")
        print(f"   💰 Retorno Total: {ret_bench:.2%} (${df_bench['Equity'].iloc[-1]:,.0f})")
        print(f"   ⚡ Sharpe Ratio:  {sharpe_bench:.2f}")
        print(f"   📉 Max Drawdown:  {dd_bench:.2%}")
        print("="*60)

        plt.figure(figsize=(12, 6))
        if not df_trend.empty:
            plt.plot(df_trend.index, df_trend["Equity"], label=f"Trend (Ret: {ret_trend:.1%})", linewidth=2)
        if not df_rev.empty:
            plt.plot(df_rev.index, df_rev["Equity"], label=f"Reversion (Ret: {ret_rev:.1%})", linewidth=2, linestyle="--")
        if not df_bench.empty:
            plt.plot(df_bench.index, df_bench["Equity"], label=f"Benchmark (Ret: {ret_bench:.1%})", color="gray", alpha=0.5)

        plt.title(f"Walk-Forward Backtest ({self.bt_config['START_DATE']} - {self.bt_config['END_DATE']})")
        plt.ylabel("Portfolio Equity ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(strategy_dir, "backtesting", "mini_models_comparison.png")
        plt.savefig(output_path)
        print(f"\n🖼️ Gráfico guardado en: {output_path}")
        plt.show()

if __name__ == "__main__":
    bf = MiniModelWalkForward()
    bf.run_comparison()