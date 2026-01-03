import os
import shutil
import logging
import concurrent.futures
import time
import polars as pl
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
QUARANTINE_DIR = os.path.join(BASE_DIR, "data", "quarantine")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MAX_MISSING_PCT = 0.15      
MAX_WORKERS = 8             

MARKET_MAP = {
    "DEFAULT": "^GSPC",
    ".MC": "^IBEX",
    ".T":  "^N225",
    ".L":  "^FTSE",
    ".DE": "^GDAXI",
    ".PA": "^FCHI",
    ".HK": "^HSI"
}

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "safety.log"), mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SafetyAudit")

def cargar_banco_calendarios():
    """
    Carga fechas y precios de cierre de los índices para alineación y detección de crashes/splits.
    """
    logger.info("Cargando Banco de Calendarios Maestros...")
    
    mapa_archivos = {}
    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".parquet"):
                ticker = file.replace(".parquet", "")
                mapa_archivos[ticker] = os.path.join(root, file)

    banco = {}
    
    indices_a_cargar = set(MARKET_MAP.values())
    
    for ticker_indice in indices_a_cargar:
        if ticker_indice in mapa_archivos:
            try:
                # IMPORTANTE: Cargamos Close también para comparar retornos (Split Logic)
                df = pl.read_parquet(mapa_archivos[ticker_indice])\
                       .select(["Date", "Close"])\
                       .unique(subset=["Date"])\
                       .sort("Date")\
                       .rename({"Close": "Close_bench"})
                
                banco[ticker_indice] = df
                logger.info(f"   ✅ Calendario cargado: {ticker_indice} ({df.height} días)")
            except Exception as e:
                logger.error(f"   ❌ Error cargando índice {ticker_indice}: {e}")
        else:
            nivel = logging.CRITICAL if ticker_indice == MARKET_MAP["DEFAULT"] else logging.WARNING
            logger.log(nivel, f"   ⚠️ Calendario no encontrado: {ticker_indice}")

    if MARKET_MAP["DEFAULT"] not in banco:
        return None
        
    return banco

def seleccionar_calendario(ticker, banco_calendarios):
    sufijo = ""
    if "." in ticker:
        sufijo = "." + ticker.split(".")[-1]
    
    indice_objetivo = MARKET_MAP.get(sufijo, MARKET_MAP["DEFAULT"])
    
    if indice_objetivo in banco_calendarios:
        return banco_calendarios[indice_objetivo]
    else:
        return banco_calendarios[MARKET_MAP["DEFAULT"]]

def auditar_activo(ruta_archivo, banco_calendarios, sector, ticker):
    try:
        # Carga perezosa del activo
        q = pl.scan_parquet(ruta_archivo)

        # OPTIMIZACIÓN: Usar collect_schema() para evitar PerformanceWarning
        try:
            schema = q.collect_schema()
        except Exception:
            return "CUARENTENA", "Archivo corrupto o ilegible"

        cols_req = ["Date", "Close", "Open", "High", "Low", "Volume"]
        # schema.names() devuelve la lista de columnas
        if not all(c in schema.names() for c in cols_req):
            return "CUARENTENA", "Faltan columnas OHLCV"

        # Obtener calendario con datos del benchmark
        calendario_maestro = seleccionar_calendario(ticker, banco_calendarios)
        q_cal = calendario_maestro.lazy() 

        # Determinar fecha inicio para no rellenar historia previa a la existencia del activo
        try:
            fecha_inicio_activo = q.select(pl.col("Date").min()).collect().item()
        except:
             return "CUARENTENA", "No se pudo leer fecha inicio (posible archivo vacío)"

        if fecha_inicio_activo is None:
             return "CUARENTENA", "Archivo sin fechas válidas"

        # Alineación (Left Join al calendario)
        q_aligned = (
            q_cal
            .filter(pl.col("Date") >= fecha_inicio_activo)
            .join(q, on="Date", how="left")
        )

        # Materializamos para chequeos lógicos complejos
        df_aligned = q_aligned.collect()

        # -----------------------------------------------------------
        # MEJORA 1: DETECCIÓN DE SPLITS NO AJUSTADOS
        # Lógica: Si Activo cae > 30% Y Benchmark cae < 25% -> Borrar Raw
        # -----------------------------------------------------------
        
        # Calculamos retornos diarios
        df_check_split = df_aligned.with_columns([
            (pl.col("Close") / pl.col("Close").shift(1) - 1).alias("ret_asset"),
            (pl.col("Close_bench") / pl.col("Close_bench").shift(1) - 1).alias("ret_bench")
        ])

        # Filtramos eventos sospechosos
        split_events = df_check_split.filter(
            (pl.col("ret_asset") < -0.30) & (pl.col("ret_bench") > -0.25)
        )

        if split_events.height > 0:
            logger.warning(f"☢️ SPLIT DETECTADO en {ticker}: Caída >30% sin crash de mercado. Eliminando RAW.")
            try:
                os.remove(ruta_archivo) # Borrado físico para forzar re-descarga
                return "SPLIT_DETECTED", "Raw eliminado por posible split no ajustado"
            except Exception as e:
                return "ERROR", f"Fallo al borrar archivo corrupto: {e}"

        # -----------------------------------------------------------
        # CHEQUEOS DE INTEGRIDAD ESTÁNDAR
        # -----------------------------------------------------------

        if df_aligned.filter(pl.col("Close") <= 0).height > 0:
            return "CUARENTENA", "Contiene precios <= 0"

        total_dias = df_aligned.height
        if total_dias == 0:
             return "CUARENTENA", "Sin intersección de fechas con el mercado"

        dias_validos = df_aligned.select(pl.col("Close").count()).item()
        dias_perdidos = total_dias - dias_validos
        pct_missing = dias_perdidos / total_dias

        if pct_missing > MAX_MISSING_PCT:
            df_aligned.write_parquet(os.path.join(QUARANTINE_DIR, f"{ticker}_HIGH_MISSING.parquet"))
            return "CUARENTENA", f"Faltan demasiados datos ({pct_missing:.1%})"

        # -----------------------------------------------------------
        # LIMPIEZA Y GUARDADO (MEJORA 2: DUMMY DE CALIDAD)
        # -----------------------------------------------------------
        
        ruta_clean_sector = os.path.join(CLEAN_DIR, sector)
        os.makedirs(ruta_clean_sector, exist_ok=True)
        ruta_final = os.path.join(ruta_clean_sector, f"{ticker}.parquet")

        df_aligned.lazy().with_columns([
            # MEJORA 2: Dummy de calidad (1=Fiable/Original, 0=Relleno/Unreliable)
            # Se calcula ANTES de hacer el forward fill
            pl.when(pl.col("Close").is_not_null()).then(1).otherwise(0).cast(pl.Int8).alias("data_quality"),
            
            # Rellenado de Gaps
            pl.col("Close").forward_fill(),
            pl.col("Open").fill_null(pl.col("Close").forward_fill()),
            pl.col("High").fill_null(pl.col("Close").forward_fill()),
            pl.col("Low").fill_null(pl.col("Close").forward_fill()),
            pl.col("Volume").fill_null(0),
            
            # Metadata
            pl.lit(sector).alias("sector"),
            pl.lit(ticker).alias("ticker")
        ])\
        .drop("Close_bench") \
        .drop_nulls(subset=["Close"])\
        .with_columns([
            pl.col("Close").cast(pl.Float32),
            pl.col("Open").cast(pl.Float32),
            pl.col("High").cast(pl.Float32),
            pl.col("Low").cast(pl.Float32),
            pl.col("Volume").cast(pl.Int64)
        ])\
        .collect()\
        .write_parquet(ruta_final)

        estado = "REPARADO" if dias_perdidos > 0 else "PERFECTO"
        return estado, f"Gaps corregidos: {dias_perdidos}"

    except Exception as e:
        return "ERROR", str(e)

def ejecutar_auditoria():
    logger.info("=== INICIO DE AUDITORÍA Y LIMPIEZA ===")
    start_time = time.time()

    # Limpieza de directorios destino
    for d in [CLEAN_DIR, QUARANTINE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    banco_calendarios = cargar_banco_calendarios()
    if not banco_calendarios:
        logger.critical("❌ ABORTANDO: No se encontró el calendario maestro (S&P 500).")
        return

    tareas = []
    logger.info(f"Escaneando {RAW_DIR}...")
    
    tickers_indices = set(MARKET_MAP.values())

    for root, _, files in os.walk(RAW_DIR):
        sector = os.path.basename(root)
        for file in files:
            if file.endswith(".parquet"):
                ticker = file.replace(".parquet", "")
                
                # Tratamiento especial para índices (se copian directos, pero casteando)
                if ticker in tickers_indices:
                    ruta_origen = os.path.join(root, file)
                    ruta_dest = os.path.join(CLEAN_DIR, sector, file)
                    os.makedirs(os.path.dirname(ruta_dest), exist_ok=True)
                    pl.read_parquet(ruta_origen).with_columns([
                        pl.col("Close").cast(pl.Float32)
                    ]).write_parquet(ruta_dest)
                    continue

                ruta_completa = os.path.join(root, file)
                tareas.append((ruta_completa, sector, ticker))

    logger.info(f"Analizando {len(tareas)} activos con {MAX_WORKERS} hilos...")

    stats = {"PERFECTO": 0, "REPARADO": 0, "CUARENTENA": 0, "SPLIT_DETECTED": 0, "ERROR": 0}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(auditar_activo, t[0], banco_calendarios, t[1], t[2]): t[2] 
            for t in tareas
        }
        
        for fut in concurrent.futures.as_completed(futures):
            tick = futures[fut]
            try:
                estado, msg = fut.result()
                stats[estado] = stats.get(estado, 0) + 1
                
                if estado == "CUARENTENA":
                    logger.warning(f"🔴 {tick}: {msg}")
                elif estado == "SPLIT_DETECTED":
                    logger.warning(f"☢️ {tick}: {msg}")
                elif estado == "ERROR":
                    logger.error(f"❌ {tick}: {msg}")
                
            except Exception as e:
                logger.error(f"Error fatal en worker {tick}: {e}")

    duracion = time.time() - start_time
    logger.info("-" * 40)
    logger.info(f"PROCESO TERMINADO EN {duracion:.2f}s")
    for k, v in stats.items():
        logger.info(f"{k}: {v}")
    logger.info("-" * 40)

if __name__ == "__main__":
    ejecutar_auditoria()