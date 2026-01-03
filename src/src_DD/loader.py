import os
import logging
import sys
import subprocess
import time
from typing import Optional, List, Dict, Union

import duckdb
import polars as pl

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [LOADER] - %(message)s')
logger = logging.getLogger("DataLoader")

class MarketLoader:
    """
    Gestor de acceso a datos basado en DuckDB (OLAP).
    Permite consultas SQL sobre todo el dataset de archivos Parquet como si fuera una sola tabla.
    """

    def __init__(self, actualizar_datos: bool = True, horas_validez: int = 24):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.clean_path = os.path.join(self.base_dir, "data", "clean")
        self.con = None # Conexión DuckDB

        # --- 1. LÓGICA DE ACTUALIZACIÓN (Igual que antes) ---
        if actualizar_datos:
            self._verificar_actualizacion(horas_validez)

        # --- 2. INICIALIZACIÓN DEL MOTOR OLAP ---
        self._init_db()

    def _verificar_actualizacion(self, horas_validez):
        ejecutar_master = False
        
        if not os.path.exists(self.clean_path) or not os.listdir(self.clean_path):
            logger.warning("⚠️ Datos no encontrados. Iniciando descarga completa...")
            ejecutar_master = True
        else:
            # Buscamos el archivo más reciente para determinar la frescura
            last_modified = 0
            for root, _, files in os.walk(self.clean_path):
                for f in files:
                    if f.endswith(".parquet"):
                        t = os.path.getmtime(os.path.join(root, f))
                        if t > last_modified: last_modified = t
            
            age_hours = (time.time() - last_modified) / 3600
            
            if age_hours > horas_validez:
                logger.info(f"⚠️ Datos obsoletos ({age_hours:.1f}h antigüedad). Actualizando...")
                ejecutar_master = True
            else:
                logger.info(f"✅ Datos vigentes ({age_hours:.1f}h). Carga inmediata.")

        if ejecutar_master:
            ruta_master = os.path.join(self.base_dir, "src", "master.py")
            try:
                logger.info("⏳ Ejecutando pipeline (Download + Safety)...")
                subprocess.run([sys.executable, ruta_master], check=True)
                logger.info("✅ Pipeline finalizado. Conectando DB.")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Error crítico en master.py: {e}")

    def _init_db(self):
        """
        Monta la base de datos DuckDB en memoria y crea la vista maestra.
        """
        try:
            self.con = duckdb.connect(database=':memory:')
            
            # Normalizamos ruta para Windows/Linux (DuckDB requiere forward slashes a veces o escapes)
            glob_path = os.path.join(self.clean_path, "**", "*.parquet").replace("\\", "/")
            
            logger.info(f"🦆 Inicializando DuckDB sobre: {glob_path}")
            
            # union_by_name=True es CRÍTICO: Permite que si algunos archivos tienen 'data_quality' y otros no, 
            # se unan igual rellenando con NULLs.
            query = f"""
            CREATE OR REPLACE VIEW market AS 
            SELECT * FROM read_parquet('{glob_path}', union_by_name=True, hive_partitioning=False)
            """
            self.con.execute(query)
            
            # Contar filas para confirmar carga
            count = self.con.execute("SELECT COUNT(*) FROM market").fetchone()[0]
            logger.info(f"✅ DB Montada. Total filas indexadas: {count:,}")
            
        except Exception as e:
            logger.critical(f"❌ Error fatal inicializando DuckDB: {e}")
            raise e

    def query(self, sql_query: str) -> pl.DataFrame:
        """
        Ejecuta SQL arbitrario y devuelve un Polars DataFrame (Zero-Copy).
        Ej: loader.query("SELECT * FROM market WHERE RSI > 70")
        """
        try:
            # .pl() convierte a Polars usando Apache Arrow (muy eficiente)
            return self.con.execute(sql_query).pl()
        except Exception as e:
            logger.error(f"Error SQL: {e}")
            return pl.DataFrame()

    def get_ticker(self, ticker: str) -> Optional[pl.DataFrame]:
        """Obtiene datos de un ticker específico usando SQL."""
        ticker = ticker.upper()
        df = self.query(f"SELECT * FROM market WHERE ticker = '{ticker}' ORDER BY Date")
        if df.is_empty():
            logger.warning(f"Ticker {ticker} no encontrado.")
            return None
        return df

    def get_sector(self, sector: str) -> pl.DataFrame:
        """Obtiene todo un sector."""
        return self.query(f"SELECT * FROM market WHERE sector = '{sector}' ORDER BY ticker, Date")

    def get_all_data(self) -> pl.DataFrame:
        """Devuelve TODO el mercado."""
        return self.query("SELECT * FROM market ORDER BY ticker, Date")

    def list_available_tickers(self) -> List[str]:
        res = self.con.execute("SELECT DISTINCT ticker FROM market ORDER BY ticker").fetchall()
        return [r[0] for r in res]

    def list_sectors(self) -> List[str]:
        res = self.con.execute("SELECT DISTINCT sector FROM market ORDER BY sector").fetchall()
        return [r[0] for r in res]

if __name__ == "__main__":
    loader = MarketLoader(actualizar_datos=False)
    
    # Test SQL
    print("\n--- TEST SQL DUCKDB ---")
    print("Top 3 sectores con más registros:")
    df_stats = loader.query("SELECT sector, COUNT(*) as count FROM market GROUP BY sector ORDER BY count DESC LIMIT 3")
    print(df_stats)