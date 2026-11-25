"""
Configuración de la API.

Los valores "mágicos" viven en un solo lugar y cada uno se puede cambiar con una
variable de entorno, sin tocar el código:

    MODEL_PATH=otro_modelo.joblib PORT=9000 uv run python api/main.py
"""

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))  # para importar proyecto_equipo_3 sin instalarlo

from proyecto_equipo_3 import modelo as md

API_VERSION = "1.0.0"
TITULO = "API de riesgo de abandono de clientes"
DESCRIPCION = (
    "Estima la probabilidad de que un cliente de telecomunicaciones abandone el servicio, "
    "para decidir a quién ofrecer una campaña de retención. "
    "Modelo: pipeline de scikit-learn (regresión logística) entrenado con Telco Customer Churn."
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", md.RUTA_MODELO))
METADATA_PATH = Path(os.getenv("METADATA_PATH", md.RUTA_METADATA))

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MAX_BATCH = int(
    os.getenv("MAX_BATCH", "100")
)  # máximo de clientes por solicitud en /predict-batch
# Probabilidad desde la cual el riesgo es "alto". En test, con 0,70 acierta ~63 % de las veces.
UMBRAL_RIESGO_ALTO = float(os.getenv("UMBRAL_RIESGO_ALTO", "0.70"))
