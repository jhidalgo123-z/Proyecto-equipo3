"""
Carga del modelo y predicción.

Separar esto de main.py permite probar la lógica del modelo sin levantar el
servidor y cambiar la forma de cargarlo sin tocar los endpoints.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from api import config
from api.config import md
from api.schemas import ClienteEntrada


class ModeloNoDisponible(RuntimeError):
    """El archivo del modelo no existe: hay que entrenarlo primero."""


class ServicioModelo:
    """Envuelve el pipeline de scikit-learn, sus metadatos y el umbral de decisión."""

    def __init__(
        self,
        ruta_modelo: Path = config.MODEL_PATH,
        ruta_metadata: Path = config.METADATA_PATH,
    ):
        if not ruta_modelo.exists():
            raise ModeloNoDisponible(
                f"No se encontró el modelo en {ruta_modelo}. "
                "Entrénalo con: uv run python scripts/train.py"
            )
        self.pipeline = joblib.load(ruta_modelo)
        self.metadata = (
            json.loads(ruta_metadata.read_text(encoding="utf-8"))
            if ruta_metadata.exists()
            else {}
        )
        self.version = str(self.metadata.get("version", "desconocida"))
        self.umbral = float(self.metadata.get("umbral", 0.5))

    def nivel_riesgo(self, probabilidad: float) -> str:
        if probabilidad < self.umbral:
            return "bajo"
        return (
            "alto"
            if probabilidad >= max(config.UMBRAL_RIESGO_ALTO, self.umbral)
            else "medio"
        )

    def predecir(self, clientes: list[ClienteEntrada]) -> list[dict]:
        """Convierte los clientes en un DataFrame y llama al pipeline.

        El pipeline hace TODO el preprocesamiento (imputar, escalar, one-hot),
        por eso aquí basta con pasarle los datos "crudos".
        """
        datos = pd.DataFrame([c.model_dump() for c in clientes])[md.FEATURES]
        probabilidades = self.pipeline.predict_proba(datos)[:, 1]
        return [
            {
                "probabilidad_abandono": round(float(p), 4),
                "abandona": bool(p >= self.umbral),
                "nivel_riesgo": self.nivel_riesgo(float(p)),
            }
            for p in probabilidades
        ]
