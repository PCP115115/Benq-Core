import logging
import time
import sys
import os

# --- SETUP DE RUTAS (Para evitar errores de "Module not found") ---
# Aseguramos que el directorio 'src' sea visible para importar los módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importamos tus módulos optimizados
try:
    import download
    import safety
    import loader
except ImportError as e:
    print(f"❌ ERROR CRÍTICO DE IMPORTACIÓN: {e}")
    print("Asegúrate de que 'download.py', 'safety.py' y 'loader.py' están en la carpeta 'src'.")
    sys.exit(1)

# Configuración del log global para el orquestador
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MASTER] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MASTER")

def main():
    start_total = time.time()
    logger.info("=========================================================")
    logger.info("🚀 INICIANDO PIPELINE QUANT: DOWNLOAD -> AUDIT -> INDEX")
    logger.info("=========================================================")

    # ---------------------------------------------------------
    # PASO 1: DESCARGA (ETL - Extract)
    # ---------------------------------------------------------
    logger.info("\n>>> [1/3] EJECUTANDO MOTOR DE DESCARGAS (download.py) <<<")
    try:
        # Aquí se ejecuta tu lógica multihilo con Backoff y requests resilientes
        download.orquestador_descargas()
    except Exception as e:
        logger.critical(f"❌ FALLO CRÍTICO EN DESCARGAS: {e}")
        # Si no bajamos datos, no tiene sentido seguir
        sys.exit(1) 

    # ---------------------------------------------------------
    # PASO 2: AUDITORÍA Y LIMPIEZA (ETL - Transform)
    # ---------------------------------------------------------
    logger.info("\n>>> [2/3] APLICANDO PROTOCOLO DE SEGURIDAD (safety.py) <<<")
    try:
        # Aquí entra Polars Lazy, Enrutamiento de Calendarios y lógica financiera
        safety.ejecutar_auditoria()
    except Exception as e:
        logger.critical(f"❌ FALLO CRÍTICO EN AUDITORÍA: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # PASO 3: VERIFICACIÓN DE CARGA (Health Check)
    # ---------------------------------------------------------
    logger.info("\n>>> [3/3] VERIFICANDO INTEGRIDAD DEL ÍNDICE (loader.py) <<<")
    try:
        # ⚠️ CRUCIAL: Pasamos actualizar_datos=False para evitar el BUCLE INFINITO.
        # Master ya acaba de actualizar los datos, solo queremos cargar el índice.
        mercado = loader.MarketLoader(actualizar_datos=False)
        
        activos = mercado.list_available_tickers()
        sectores = mercado.list_sectors()
        total_activos = len(activos)

        logger.info("-" * 40)
        logger.info(f"✅ PIPELINE COMPLETADO EXITOSAMENTE.")
        logger.info(f"📊 INFORME FINAL DE DISPONIBILIDAD:")
        logger.info(f"   - Total Activos Limpios: {total_activos}")
        logger.info(f"   - Sectores Indexados:    {len(sectores)}")
        logger.info(f"   - Lista de Sectores:     {', '.join(sectores)}")
        
        if total_activos > 0:
            # Pequeña prueba de humo: cargar el primer activo para asegurar que el loader responde
            ejemplo = activos[0]
            df = mercado.get_ticker(ejemplo)
            if df is not None:
                logger.info(f"   - Test de Lectura ({ejemplo}): OK ({df.height} filas)")
        
    except Exception as e:
        logger.error(f"⚠️ El pipeline terminó, pero el Loader dio error al verificar: {e}")

    # ---------------------------------------------------------
    # RESUMEN DE TIEMPO
    # ---------------------------------------------------------
    duracion = time.time() - start_total
    logger.info("=========================================================")
    logger.info(f"🏁 TIEMPO TOTAL DE EJECUCIÓN: {duracion:.2f} segundos")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
