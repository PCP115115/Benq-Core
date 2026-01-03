import os
import logging
import socket
import concurrent.futures
import time
import random
from datetime import datetime, timedelta

import yfinance as yf
import polars as pl
import pandas as pd 

try:
    from tickers import SECTOR_TICKERS
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    try:
        from tickers import SECTOR_TICKERS
    except ImportError:
        SECTOR_TICKERS = {"TECNOLOGIA": ["AAPL", "MSFT", "NVDA"]}


BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


FECHA_INICIO_DEFECTO = "2010-01-01" 
FECHA_FIN_GLOBAL = datetime.now().strftime('%Y-%m-%d')

MAX_WORKERS = 5       
TIMEOUT_SECONDS = 20  
MAX_RETRIES = 3       

socket.setdefaulttimeout(TIMEOUT_SECONDS)

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (%(threadName)s) - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "download.log"), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def descargar_con_backoff(ticker, start, end):
    for intento in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, threads=False, auto_adjust=True)
            
            if not df.empty:
                return df
            
            if intento < MAX_RETRIES - 1:
                sleep_time = (2 ** intento) + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.warning(f"Warning {ticker} (Intento {intento+1}): {e}")
            
    return pd.DataFrame()

def obtener_ultima_fecha(df_polars: pl.DataFrame) -> datetime:
    try:
        col_fecha = "Date" if "Date" in df_polars.columns else "index"
        if col_fecha not in df_polars.columns:
            return None
        
        max_date = df_polars.select(pl.col(col_fecha).max()).item()
        
        if isinstance(max_date, str):
            return datetime.strptime(max_date, '%Y-%m-%d')
        if isinstance(max_date, (datetime, pd.Timestamp)):
            return max_date
        return pd.to_datetime(max_date).to_pydatetime()
    except:
        return None

def procesar_activo(ticker: str, sector: str):
    try:
        ruta_sector = os.path.join(BASE_DATA_PATH, sector)
        os.makedirs(ruta_sector, exist_ok=True)
        ruta_final = os.path.join(ruta_sector, f"{ticker}.parquet")

        df_existente = None
        fecha_inicio_descarga = FECHA_INICIO_DEFECTO
        modo_incremental = False

        if os.path.exists(ruta_final):
            try:
                df_existente = pl.read_parquet(ruta_final)
                ultima_fecha = obtener_ultima_fecha(df_existente)

                if ultima_fecha:
                    fecha_limite = datetime.now() - timedelta(days=1)
                    
                    if ultima_fecha.date() >= fecha_limite.date():
                        logger.info(f"[{ticker}] Datos frescos ({ultima_fecha.date()}). Omitiendo.")
                        return 

                    fecha_inicio_descarga = (ultima_fecha + timedelta(days=1)).strftime('%Y-%m-%d')
                    modo_incremental = True
                    logger.info(f"[{ticker}] Actualizando... (Delta desde {fecha_inicio_descarga})")
                else:
                    logger.warning(f"[{ticker}] Archivo ilegible. Re-descargando completo.")
            except Exception:
                logger.error(f"[{ticker}] Archivo corrupto. Re-descargando completo.")
                df_existente = None

        df_pandas_nuevo = descargar_con_backoff(ticker, fecha_inicio_descarga, FECHA_FIN_GLOBAL)

        if df_pandas_nuevo.empty:
            if modo_incremental:
                logger.info(f"[{ticker}] Sin datos nuevos disponibles hoy.")
                return
            else:
                logger.warning(f"[{ticker}] No se encontraron datos (Ticker inválido o deslistado).")
                return

        if isinstance(df_pandas_nuevo.columns, pd.MultiIndex):
            df_pandas_nuevo.columns = df_pandas_nuevo.columns.droplevel(1)
        
        df_pandas_nuevo.reset_index(inplace=True)
        df_pandas_nuevo.columns = [str(col).strip().replace(" ", "_") for col in df_pandas_nuevo.columns]
        df_pandas_nuevo = df_pandas_nuevo.loc[:, ~df_pandas_nuevo.columns.duplicated()]

        df_polars_nuevo = pl.from_pandas(df_pandas_nuevo)
        
        df_polars_nuevo = df_polars_nuevo.with_columns([
            pl.lit(sector).alias("sector"),
            pl.lit(ticker).alias("ticker"),
            pl.col("Date").cast(pl.Datetime) if "Date" in df_polars_nuevo.columns else pl.col("index").cast(pl.Datetime).alias("Date")
        ])

        df_final = df_polars_nuevo
        
        if modo_incremental and df_existente is not None:
            try:
                df_final = pl.concat([df_existente, df_polars_nuevo], how="vertical_relaxed")
                df_final = df_final.unique(subset=["Date", "ticker"], keep="last").sort("Date")
            except Exception as e:
                logger.error(f"[{ticker}] Error en fusión (Merge): {e}. Sobrescribiendo con nuevos datos.")
                df_final = df_polars_nuevo 

        df_final.write_parquet(ruta_final)
        
        estado = "ACTUALIZADO" if modo_incremental else "CREADO"
        logger.info(f"[OK] {ticker} {estado} | Filas: {df_final.height} | Fin: {obtener_ultima_fecha(df_final).date()}")

    except Exception as e:
        logger.error(f"[ERROR CRITICO] {ticker}: {e}")

def orquestador_descargas():
    start_time = time.time()
    logger.info("--- INICIO DE ACTUALIZACIÓN DE DATOS DE MERCADO ---")
    
    tareas = []
    for sector, lista_tickers in SECTOR_TICKERS.items():
        for ticker in lista_tickers:
            tareas.append((ticker, sector))
    
    random.shuffle(tareas)

    logger.info(f"Procesando {len(tareas)} activos con {MAX_WORKERS} hilos.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(procesar_activo, t, s): t for t, s in tareas}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            pass

    duracion = time.time() - start_time
    logger.info(f"--- PROCESO COMPLETADO EN {duracion:.2f} SEGUNDOS ---")

if __name__ == "__main__":
    orquestador_descargas()
