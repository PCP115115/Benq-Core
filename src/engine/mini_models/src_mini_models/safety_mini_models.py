import os
import polars as pl
import logging
import joblib

logger = logging.getLogger("SafetyModels")

def get_model_path(base_dir: str, ticker: str, model_type: str) -> str:
    """Genera la ruta segura para guardar/cargar modelos."""
    ticker_clean = ticker.replace("^", "").replace("=", "")
    path = os.path.join(base_dir, ticker_clean, f"{model_type}.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def validate_input_data(df: pl.DataFrame, required_features: list) -> bool:
    """
    Verifica integridad de datos antes del entrenamiento/inferencia.
    Retorna True si es válido, levanta excepción si no.
    """
    if df.is_empty():
        raise ValueError("DataFrame vacío.")

    # 1. Verificar columnas
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan features requeridas: {missing}")

    # 2. Verificar Infs/Nulls críticos en features
    # (El pipeline robusto ya debería haber limpiado, pero doble check)
    null_counts = df.select([pl.col(c).null_count() for c in required_features]).sum(axis=1).item()
    if null_counts > 0:
        logger.warning(f"⚠️ Datos contienen {null_counts} valores nulos en features. Se eliminarán filas.")
        return False # Indica que el caller debe hacer drop_nulls()

    return True

def check_models_exist(base_dir: str, ticker: str, model_types: list) -> bool:
    """Verifica si todos los modelos existen para un ticker."""
    return all(os.path.exists(get_model_path(base_dir, ticker, mt)) for mt in model_types)