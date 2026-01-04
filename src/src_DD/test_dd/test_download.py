import unittest
import os
import sys
import shutil
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Asumiendo estructura: /tests/test_download.py -> /src/download.py
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

# Importamos el módulo completo para poder mockear sus componentes
import download 
from download import procesar_activo, FECHA_INICIO_DEFECTO

class TestDescargaIncremental(unittest.TestCase):

    def setUp(self):
        """Preparación del entorno aislado para cada test."""
        self.ticker_prueba = "TEST_TICKER"
        self.sector_prueba = "Test_Sector"
        
        # Directorio temporal aislado
        self.root_temp = os.path.join(current_dir, "temp_test_env")
        self.ruta_raw_temp = os.path.join(self.root_temp, "raw")
        self.ruta_archivo_final = os.path.join(self.ruta_raw_temp, self.sector_prueba, f"{self.ticker_prueba}.parquet")
        
        os.makedirs(self.ruta_raw_temp, exist_ok=True)

        # Mock del path global: Forzamos a que el script use nuestra carpeta temporal
        self.patcher_path = patch('download.BASE_DATA_PATH', self.ruta_raw_temp)
        self.patcher_path.start()

    def tearDown(self):
        """Limpieza post-test."""
        self.patcher_path.stop()
        if os.path.exists(self.root_temp):
            shutil.rmtree(self.root_temp, ignore_errors=True)

    def _generar_dummy_data(self, start_date, end_date):
        """Helper para crear datos falsos de mercado."""
        fechas = pd.date_range(start=start_date, end=end_date, freq='D') 
        
        data = {
            'Open': [100.0] * len(fechas),
            'High': [105.0] * len(fechas),
            'Low': [95.0] * len(fechas),
            'Close': [102.0] * len(fechas),
            'Volume': [1000] * len(fechas)
        }
        df = pd.DataFrame(data, index=fechas)
        df.index.name = "Date"
        return df

    # --- TEST 1: INTEGRACIÓN REAL (Cold Start) ---
    def test_01_descarga_real_cold_start(self):
        """
        Prueba E2E real contra Yahoo Finance. 
        Verifica que si no hay archivo, descarga y guarda con el formato correcto.
        """
        print(f"\n🧪 [Test 1] Integración Real: Descarga Cold Start (AAPL)...")
        
        # Usamos un ticker real para este test
        ticker_real = "AAPL"
        ruta_real = os.path.join(self.ruta_raw_temp, self.sector_prueba, f"{ticker_real}.parquet")
        
        procesar_activo(ticker_real, self.sector_prueba)

        self.assertTrue(os.path.exists(ruta_real), "❌ El archivo Parquet no se generó.")
        
        df = pl.read_parquet(ruta_real)
        self.assertFalse(df.is_empty(), "❌ El DataFrame descargado está vacío.")
        self.assertIn("Close", df.columns)
        self.assertIn("Date", df.columns)
        print("   ✅ Descarga real y guardado exitoso.")

    # --- TEST 2: LÓGICA INCREMENTAL (Mocked) ---
    @patch('download.yf.download') 
    def test_02_logica_incremental_append(self, mock_yf):
        """
        Simula que ya tenemos datos hasta hace 5 días.
        Verifica que el script SÓLO pide los 5 días faltantes y los une correctamente.
        """
        print(f"\n🧪 [Test 2] Lógica Incremental: Append de datos nuevos...")

        # 1. CREAR ESTADO INICIAL (Datos viejos)
        fecha_fin_vieja = datetime.now() - timedelta(days=10)
        df_old_pd = self._generar_dummy_data("2020-01-01", fecha_fin_vieja)
        
        # Guardamos el archivo "existente"
        os.makedirs(os.path.dirname(self.ruta_archivo_final), exist_ok=True)
        df_old_pl = pl.from_pandas(df_old_pd.reset_index()).with_columns([
            pl.lit(self.sector_prueba).alias("sector"),
            pl.lit(self.ticker_prueba).alias("ticker")
        ])
        df_old_pl.write_parquet(self.ruta_archivo_final)
        
        filas_iniciales = df_old_pl.height

        # 2. PREPARAR MOCK DE DATOS NUEVOS
        # Simulamos que Yahoo devuelve los datos desde T-9 hasta hoy
        fecha_inicio_nuevos = fecha_fin_vieja + timedelta(days=1)
        df_new_pd = self._generar_dummy_data(fecha_inicio_nuevos, datetime.now())
        mock_yf.return_value = df_new_pd

        # 3. EJECUTAR SCRIPT
        procesar_activo(self.ticker_prueba, self.sector_prueba)

        # 4. VERIFICACIONES
        # A) ¿Llamó a yfinance con la fecha correcta?
        args, kwargs = mock_yf.call_args
        fecha_llamada = kwargs.get('start')
        
        # Debería pedir desde el día siguiente al que teníamos
        expected_start = (fecha_fin_vieja + timedelta(days=1)).strftime('%Y-%m-%d')
        self.assertEqual(fecha_llamada, expected_start, f"❌ Fecha de inicio incorrecta. Esperada: {expected_start}, Real: {fecha_llamada}")

        # B) ¿Se fusionaron los datos?
        df_final = pl.read_parquet(self.ruta_archivo_final)
        self.assertGreater(df_final.height, filas_iniciales, "❌ No se añadieron filas nuevas.")
        
        # Validar continuidad: Fecha minima vieja y fecha maxima nueva
        fechas = df_final["Date"].dt.date()
        self.assertEqual(fechas.min(), pd.to_datetime("2020-01-01").date())
        self.assertGreater(fechas.max(), fecha_fin_vieja.date())
        
        print(f"   ✅ Correcto: Se pidieron datos desde {expected_start} y se fusionaron ({filas_iniciales} -> {df_final.height} filas).")

    # --- TEST 3: EFICIENCIA (Datos Frescos) ---
    @patch('download.yf.download')
    def test_03_datos_frescos_no_descarga(self, mock_yf):
        """
        Simula que el archivo ya está actualizado hasta ayer.
        Verifica que NO se llama a yfinance.
        """
        print(f"\n🧪 [Test 3] Eficiencia: Archivo actualizado...")

        # 1. Crear archivo actualizado (hasta ayer)
        fecha_ayer = datetime.now() - timedelta(days=1) # Asumimos ayer como dato más reciente posible
        df_fresh = self._generar_dummy_data("2024-01-01", fecha_ayer)
        
        os.makedirs(os.path.dirname(self.ruta_archivo_final), exist_ok=True)
        df_pl = pl.from_pandas(df_fresh.reset_index()).with_columns([
            pl.lit(self.sector_prueba).alias("sector"),
            pl.lit(self.ticker_prueba).alias("ticker")
        ])
        df_pl.write_parquet(self.ruta_archivo_final)

        # 2. EJECUTAR SCRIPT
        procesar_activo(self.ticker_prueba, self.sector_prueba)

        # 3. VERIFICAR QUE NO SE LLAMÓ A YAHOO
        mock_yf.assert_not_called()
        print("   ✅ Correcto: Se detectaron datos frescos y se omitió la descarga.")

if __name__ == '__main__':
    unittest.main(verbosity=2)