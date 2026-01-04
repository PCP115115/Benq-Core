import unittest
import os
import sys
import shutil
import polars as pl
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

import safety
from safety import auditar_activo, seleccionar_calendario

class TestSafety(unittest.TestCase):

    def setUp(self):
        self.root_temp = os.path.join(current_dir, "temp_safety_env")
        self.raw_dir = os.path.join(self.root_temp, "raw")
        self.clean_dir = os.path.join(self.root_temp, "clean")
        self.quarantine_dir = os.path.join(self.root_temp, "quarantine")
        
        os.makedirs(os.path.join(self.raw_dir, "TestSector"), exist_ok=True)
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)

        # 1. CREAR BANCO DE CALENDARIOS SIMULADO (Suficientes días para no activar Gaps > 15%)
        # Creamos 30 días de historia
        self.fechas_globales = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        
        bench_prices = [100.0 + i for i in range(30)]
        
        df_cal = pl.DataFrame({
            "Date": self.fechas_globales,
            "Close_bench": bench_prices 
        }).sort("Date")
        
        self.banco_calendarios = {
            "^GSPC": df_cal,
            "^IBEX": df_cal
        }

        self.old_clean = safety.CLEAN_DIR
        self.old_quarantine = safety.QUARANTINE_DIR
        safety.CLEAN_DIR = self.clean_dir
        safety.QUARANTINE_DIR = self.quarantine_dir

    def tearDown(self):
        if os.path.exists(self.root_temp):
            try:
                shutil.rmtree(self.root_temp)
            except Exception:
                pass
        safety.CLEAN_DIR = self.old_clean
        safety.QUARANTINE_DIR = self.old_quarantine

    def crear_parquet_falso(self, ticker, dias_presentes, valores):
        count = len(dias_presentes)
        vals = valores if len(valores) == count else [valores[0]] * count
        
        df = pl.DataFrame({
            "Date": dias_presentes,
            "Close": vals, "Open": vals, "High": vals, "Low": vals,
            "Volume": [1000] * count
        }).with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Date").cast(pl.Datetime)
        ])
        
        ruta = os.path.join(self.raw_dir, "TestSector", f"{ticker}.parquet")
        df.write_parquet(ruta)
        return ruta

    def test_split_detection_logic(self):
        print("\n🧪 Test Safety: Protección contra Splits...")
        
        fechas = self.fechas_globales[:2]
        precios_split = [100.0, 40.0] 
        
        ruta = self.crear_parquet_falso("SPLITTER", fechas, precios_split)
        
        with patch("os.remove") as mock_remove:
            estado, msg = auditar_activo(ruta, self.banco_calendarios, "TestSector", "SPLITTER")
            
            self.assertEqual(estado, "SPLIT_DETECTED", "❌ No detectó el Split masivo")
            self.assertIn("Raw eliminado", msg)
            mock_remove.assert_called_with(ruta)

    def test_quality_dummy_on_repair(self):
        print("🧪 Test Safety: Dummy de Calidad (Gap Filling)...")
        
        # Para que no salte la cuarentena (Max 15% gaps), necesitamos suficientes días.
        # Usamos 20 días totales. Huecos en índices 1 y 2.
        # Dias totales: 20. Faltantes: 2. Ratio: 10% (OK)
        
        dias_total = self.fechas_globales[:20]
        # Quitamos el día 2 y 3 (índices 1 y 2)
        dias_presentes = [d for i, d in enumerate(dias_total) if i not in [1, 2]]
        
        vals = [10.0] * len(dias_presentes)
        
        ruta = self.crear_parquet_falso("GAP_TEST", dias_presentes, vals)
        
        estado, msg = auditar_activo(ruta, self.banco_calendarios, "TestSector", "GAP_TEST")
        self.assertEqual(estado, "REPARADO", f"Falló status. Msg: {msg}")
        
        df_clean = pl.read_parquet(os.path.join(self.clean_dir, "TestSector", "GAP_TEST.parquet"))
        self.assertIn("data_quality", df_clean.columns)
        
        # Verificar que los huecos (que ahora existen por el fill) tienen calidad 0
        df_filled = df_clean.filter(pl.col("data_quality") == 0)
        self.assertTrue(df_filled.height > 0, "❌ No se marcaron datos rellenos con quality=0")
        self.assertEqual(df_filled.height, 2, "Debería haber exactamente 2 días rellenados")

    def test_activo_perfecto(self):
        ruta = self.crear_parquet_falso("PERFECT", self.fechas_globales[:10], [100.0]*10)
        estado, msg = auditar_activo(ruta, self.banco_calendarios, "TestSector", "PERFECT")
        self.assertEqual(estado, "PERFECTO")
        
        df = pl.read_parquet(os.path.join(self.clean_dir, "TestSector", "PERFECT.parquet"))
        self.assertEqual(df["data_quality"].min(), 1)

    def test_high_missing_quarantine(self):
        # 1 día presente de 30 posibles -> ratio missing > 90%
        fechas = [self.fechas_globales[0]]
        ruta = self.crear_parquet_falso("MISSING", fechas, [100.0])
        estado, msg = auditar_activo(ruta, self.banco_calendarios, "TestSector", "MISSING")
        self.assertEqual(estado, "CUARENTENA")

if __name__ == '__main__':
    unittest.main()