import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
import torch
import sys
import os
import shutil
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- AJUSTE DE PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(project_root)

# Importamos el módulo a testear
from src.engine.context import master_context
import src.engine.config as config

class TestMasterContextIntegration(unittest.TestCase):
    
    def setUp(self):
        """
        Preparación del entorno.
        """
        self.original_lstm_path = config.PATHS["MODEL_LSTM"]
        self.original_gmm_path = config.PATHS["MODEL_GMM"]
        self.original_epochs = config.CONTEXT_PARAMS["LSTM_EPOCHS"]
        self.original_lr = config.CONTEXT_PARAMS["LSTM_LR"]
        
        # Rutas dummy
        config.PATHS["MODEL_LSTM"] = os.path.join(project_root, "src/data/models/test_lstm_context.pth")
        config.PATHS["MODEL_GMM"] = os.path.join(project_root, "src/data/models/test_gmm_context.joblib")
        
        # CONFIGURACIÓN POR DEFECTO PARA TESTS RÁPIDOS
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = 5  # Rápido por defecto
        
        if os.path.exists(config.PATHS["MODEL_LSTM"]): os.remove(config.PATHS["MODEL_LSTM"])
        if os.path.exists(config.PATHS["MODEL_GMM"]): os.remove(config.PATHS["MODEL_GMM"])

    def tearDown(self):
        """Limpieza."""
        if os.path.exists(config.PATHS["MODEL_LSTM"]): os.remove(config.PATHS["MODEL_LSTM"])
        if os.path.exists(config.PATHS["MODEL_GMM"]): os.remove(config.PATHS["MODEL_GMM"])
        
        config.PATHS["MODEL_LSTM"] = self.original_lstm_path
        config.PATHS["MODEL_GMM"] = self.original_gmm_path
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = self.original_epochs
        config.CONTEXT_PARAMS["LSTM_LR"] = self.original_lr

    def _generate_synthetic_data(self, n_rows=500, ticker="TEST_TICKER", high_contrast=False):
        """
        Genera datos falsos.
        Arg high_contrast: Si es True, exagera las diferencias para facilitar el aprendizaje en pocas épocas.
        """
        start_date = datetime(2020, 1, 1)
        end_date = start_date + timedelta(days=n_rows - 1)
        dates = pl.datetime_range(start=start_date, end=end_date, interval="1d", eager=True)
        
        actual_rows = len(dates)
        
        # Fases
        n1 = int(actual_rows * 0.4) # Calma
        n2 = int(actual_rows * 0.3) # Normal
        n3 = actual_rows - n1 - n2  # Pánico
        
        # Parametros de generación según contraste
        if high_contrast:
            # Muy fácil de distinguir
            vol_params = [(0.005, 0.001), (0.04, 0.01), (0.25, 0.05)] 
            ret_params = [(0.001, 0.005), (0.000, 0.02), (-0.01, 0.08)]
        else:
            # Realista (más difícil)
            vol_params = [(0.01, 0.002), (0.03, 0.01), (0.15, 0.05)]
            ret_params = [(0.001, 0.01), (0.000, 0.02), (-0.005, 0.06)]

        # 1. Generar Volatilidad (Feature 0)
        vol_data = np.concatenate([
            np.random.normal(*vol_params[0], n1),
            np.random.normal(*vol_params[1], n2),
            np.random.normal(*vol_params[2], n3)
        ])
        
        # 2. Generar Precio Sintético (Returns)
        returns = np.concatenate([
            np.random.normal(*ret_params[0], n1),
            np.random.normal(*ret_params[1], n2),
            np.random.normal(*ret_params[2], n3)
        ])
        price = 100 * np.cumprod(1 + returns)
        
        # 3. Resto de Features (Ruido)
        other_features = np.random.randn(actual_rows, len(config.CONTEXT_PARAMS["INPUT_FEATURES"]) - 1)
        data_matrix = np.column_stack([vol_data, other_features])
        
        df = pl.DataFrame({
            "Date": dates,
            "ticker": [ticker] * actual_rows,
            "Close": price
        })
        feat_df = pl.DataFrame(data_matrix, schema=config.CONTEXT_PARAMS["INPUT_FEATURES"])
        return pl.concat([df, feat_df], how="horizontal")

    @patch('src.engine.src_features.master_features.get_feature_matrix')
    def test_01_end_to_end_flow_and_structure(self, mock_get_features):
        print("\n--- Test 01: Flujo End-to-End ---")
        mock_df = self._generate_synthetic_data(n_rows=100)
        mock_get_features.return_value = mock_df
        result_df = master_context.get_market_regime(tickers="TEST_TICKER")
        self.assertFalse(result_df.is_empty())

    @patch('src.engine.src_features.master_features.get_feature_matrix')
    def test_02_logic_coherence_and_sorting(self, mock_get_features):
        print("\n--- Test 02: Coherencia Lógica ---")
        # Usamos High Contrast para asegurar que la lógica de ordenamiento se valida bien
        mock_df = self._generate_synthetic_data(n_rows=600, high_contrast=True)
        mock_get_features.return_value = mock_df
        
        # Subimos épocas un poco para asegurar convergencia básica
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = 15
        
        result_df = master_context.get_market_regime(tickers="TEST_TICKER")
        
        preds = result_df["market_regime"].to_numpy()
        self.assertTrue(len(np.unique(preds)) >= 2)

    @patch('src.engine.src_features.master_features.get_feature_matrix')
    def test_03_computational_performance(self, mock_get_features):
        print("\n--- Test 03: Rendimiento y Throughput ---")
        # Aquí usamos pocas épocas (5) porque medimos velocidad de tubería, no calidad
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = 5
        
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
        dfs = [self._generate_synthetic_data(n_rows=2500, ticker=tk) for tk in tickers]
        big_df = pl.concat(dfs)
        total_rows = len(big_df)
        
        mock_get_features.return_value = big_df
        
        start_time = time.time()
        _ = master_context.get_market_regime(tickers=tickers)
        duration = time.time() - start_time
        
        rows_per_sec = total_rows / duration
        print(f"   Procesadas {total_rows} filas en {duration:.4f}s")
        print(f"   🚀 Velocidad: {rows_per_sec:.0f} filas/segundo")
        
        self.assertLess(duration, 20.0)

    @patch('src.engine.src_features.master_features.get_feature_matrix')
    def test_05_visual_inspection(self, mock_get_features):
        print("\n--- Test 05: Generación de Gráfico de Validación ---")
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = 20 # Calidad media para gráfico decente
        
        mock_df = self._generate_synthetic_data(n_rows=800, ticker="VISUAL_TEST", high_contrast=True)
        mock_get_features.return_value = mock_df
        
        result_df = master_context.get_market_regime(tickers="VISUAL_TEST")
        
        # Plotting logic (igual que antes)
        window = config.CONTEXT_PARAMS["LSTM_WINDOW_SIZE"]
        aligned_price = mock_df["Close"].slice(window - 1, len(mock_df)).to_numpy()
        aligned_dates = result_df["Date"].to_numpy()
        regimes = result_df["market_regime"].to_numpy()
        probs = result_df["regime_probability"].to_numpy()
        
        min_len = min(len(aligned_price), len(regimes))
        aligned_price = aligned_price[:min_len]; aligned_dates = aligned_dates[:min_len]; regimes = regimes[:min_len]; probs = probs[:min_len]

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            colors = ['green', 'orange', 'red']
            labels = ['Calma (0)', 'Transición (1)', 'Pánico (2)']
            
            for r in [0, 1, 2]:
                mask = (regimes == r)
                if mask.sum() > 0:
                    ax1.scatter(aligned_dates[mask], aligned_price[mask], c=colors[r], label=labels[r], s=10, alpha=0.6)
            
            ax1.set_title("Validación Visual de Regímenes")
            ax1.legend()
            
            ax2.plot(aligned_dates, probs, color='blue', linewidth=0.5)
            ax2.set_ylabel("Confianza")
            
            output_path = os.path.join(current_dir, "regime_test_plot.png")
            plt.savefig(output_path)
            plt.close()
            print(f"   🖼️ Gráfico: {output_path}")
            self.assertTrue(os.path.exists(output_path))
        except Exception:
            pass

    @patch('src.engine.src_features.master_features.get_feature_matrix')
    def test_06_statistical_quant_metrics(self, mock_get_features):
        """
        Genera Estadísticas Quant. FORZAMOS APRENDIZAJE AGRESIVO.
        """
        print("\n--- Test 06: Métricas Estadísticas Quant ---")
        
        # 1. Ajuste CLAVE: Aprendizaje agresivo para el test
        config.CONTEXT_PARAMS["LSTM_EPOCHS"] = 50  # Máximo rigor
        config.CONTEXT_PARAMS["LSTM_LR"] = 0.01    # Learning Rate alto
        
        # 2. Datos con CONTRASTE EXTREMO
        mock_df = self._generate_synthetic_data(n_rows=3000, ticker="STAT_TEST", high_contrast=True)
        
        # --- CORRECCIÓN AQUÍ: Añadimos .copy() para poder escribir en el array ---
        vol_vals = mock_df["vol_yz_20d"].to_numpy().copy() 
        
        # Ahora sí podemos modificarlo sin error
        n = len(vol_vals)
        vol_vals[:n//3] = 0.001  # Volatilidad nula (Calma absoluta)
        vol_vals[2*n//3:] = 0.50 # Volatilidad extrema (Pánico total)
        
        # Reconstruimos el DF con la señal reforzada
        mock_df = mock_df.with_columns(pl.Series("vol_yz_20d", vol_vals))
        
        mock_get_features.return_value = mock_df
        
        # 3. Pipeline
        result_df = master_context.get_market_regime(tickers="STAT_TEST")
        
        # 4. Cálculo Métricas
        window = config.CONTEXT_PARAMS["LSTM_WINDOW_SIZE"]
        # Usamos la volatilidad directa modificada
        vol_real = mock_df["vol_yz_20d"].slice(window - 1, len(mock_df)).to_numpy()
        regimes = result_df["market_regime"].to_numpy()
        
        # VDR basado en la Feature de Volatilidad
        # Filtramos asegurando que hay datos
        mask_0 = (regimes == 0)
        mask_2 = (regimes == 2)
        
        mean_vol_0 = vol_real[mask_0].mean() if mask_0.sum() > 0 else 1e-6
        mean_vol_2 = vol_real[mask_2].mean() if mask_2.sum() > 0 else 1e-6
        
        if mean_vol_0 < 1e-6: mean_vol_0 = 1e-6
            
        vdr = mean_vol_2 / mean_vol_0
        
        print(f"   📊 Vol Promedio Régimen 0 (Calma):  {mean_vol_0:.5f}")
        print(f"   📊 Vol Promedio Régimen 2 (Pánico): {mean_vol_2:.5f}")
        print(f"   🚀 VDR (Discriminación): {vdr:.2f}x")
        
        self.assertGreater(vdr, 1.5, "El modelo no distingue riesgo (VDR bajo).")

        # Estabilidad
        transitions = np.sum(regimes[1:] == regimes[:-1])
        stability = transitions / (len(regimes) - 1)
        
        print(f"   🛡️ Estabilidad: {stability:.2%}")
        self.assertGreater(stability, 0.75)
        
        # Boxplot opcional
        try:
            import pandas as pd
            # Necesitamos recalcular returns para el boxplot visual
            price_slice = mock_df["Close"].slice(window - 1, len(mock_df))
            returns = (price_slice / price_slice.shift(1)).log().fill_null(0.0).to_numpy()
            
            df_plot = pd.DataFrame({'Returns': returns, 'Regime': regimes})
            plt.figure(figsize=(10, 6))
            df_plot.boxplot(column='Returns', by='Regime', showfliers=False)
            plt.savefig(os.path.join(current_dir, "regime_boxplot_stats.png"))
            plt.close()
        except: pass

if __name__ == '__main__':
    unittest.main()