import unittest
import os
import sys
import shutil
import polars as pl
from unittest.mock import patch

# --- SETUP DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

from loader import MarketLoader

class TestLoader(unittest.TestCase):

    def setUp(self):
        """Creamos un entorno con datos falsos limpios para pruebas de DuckDB"""
        self.root_temp = os.path.join(current_dir, "temp_loader_env")
        self.clean_dir = os.path.join(self.root_temp, "data", "clean")
        
        # Estructura de carpetas
        os.makedirs(os.path.join(self.clean_dir, "Tech"), exist_ok=True)
        
        # Parquet falso 1
        self.df_tech = pl.DataFrame({
            "Date": [1, 2, 3], 
            "Close": [10.0, 11.0, 12.0],
            "ticker": ["TEST_TECH"]*3,
            "sector": ["Tech"]*3
        })
        self.df_tech.write_parquet(os.path.join(self.clean_dir, "Tech", "TEST_TECH.parquet"))

        # Parquet falso 2 (Otro sector)
        os.makedirs(os.path.join(self.clean_dir, "Energy"), exist_ok=True)
        self.df_energy = pl.DataFrame({
            "Date": [1, 2], 
            "Close": [50.0, 51.0],
            "ticker": ["TEST_NRG"]*2,
            "sector": ["Energy"]*2
        })
        self.df_energy.write_parquet(os.path.join(self.clean_dir, "Energy", "TEST_NRG.parquet"))

    def tearDown(self):
        if os.path.exists(self.root_temp):
            try:
                shutil.rmtree(self.root_temp)
            except:
                pass

    def get_loader_on_temp(self):
        """Helper para inicializar el loader apuntando a la carpeta temporal"""
        # Inicializamos con False para que no intente ejecutar master.py
        loader = MarketLoader(actualizar_datos=False)
        # Sobreescribimos la ruta
        loader.clean_path = self.clean_dir
        # Re-inicializamos DuckDB en la nueva ruta
        loader._init_db()
        return loader

    def test_duckdb_initialization(self):
        """Verifica que DuckDB indexa correctamente los archivos"""
        print("\n🧪 Test Loader: Inicialización DuckDB...")
        loader = self.get_loader_on_temp()
        
        # Contamos filas totales (3 de Tech + 2 de Energy = 5)
        res = loader.query("SELECT COUNT(*) FROM market")
        count = res.item()
        self.assertEqual(count, 5, "DuckDB no contó correctamente todas las filas")
        print("   ✅ Indexado correcto.")

    def test_sql_query_filtering(self):
        """Verifica que podemos lanzar SQL arbitrario"""
        print("🧪 Test Loader: SQL Query...")
        loader = self.get_loader_on_temp()
        
        # Filtro por sector
        df = loader.query("SELECT * FROM market WHERE sector = 'Energy'")
        self.assertEqual(df.height, 2)
        self.assertEqual(df["ticker"][0], "TEST_NRG")
        print("   ✅ SQL Filtering funciona.")

    def test_get_ticker(self):
        """Verifica el método helper get_ticker"""
        print("🧪 Test Loader: get_ticker()...")
        loader = self.get_loader_on_temp()
        
        df = loader.get_ticker("TEST_TECH")
        self.assertIsNotNone(df)
        self.assertEqual(df.height, 3)
        self.assertEqual(df["Close"][0], 10.0)
        print("   ✅ get_ticker funciona.")

    @patch('loader.subprocess.run')
    def test_trigger_update_logic(self, mock_subprocess):
        """Verifica que la lógica de actualización sigue viva"""
        print("🧪 Test Loader: Trigger Update...")
        
        # Forzamos actualización pasando True
        # Nota: Al fallar la ruta real (no existe en test), saltará warning,
        # pero lo que nos importa es si llama a subprocess.
        try:
            _ = MarketLoader(actualizar_datos=True)
        except:
            pass
        
        # Debe haber intentado llamar a master.py
        self.assertTrue(mock_subprocess.called)
        print("   ✅ Lógica de actualización intacta.")

if __name__ == '__main__':
    unittest.main()