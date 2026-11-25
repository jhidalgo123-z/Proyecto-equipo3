"""
API REST para el modelo de abandono de clientes.

Levantar el servidor (desde la raíz del repo)::

    uv run uvicorn api.main:app --reload
    # o
    uv run python api/main.py

Luego abre http://127.0.0.1:8000/docs para probarla desde el navegador.

Endpoints
---------
GET  /health          → ¿está viva la API? ¿cargó el modelo?
GET  /model-info      → versión, fecha, variables, umbral y métricas del modelo
POST /predict         → riesgo de abandono de UN cliente
POST /predict-batch   → riesgo de VARIOS clientes (máximo 100)

Códigos de error:
    422 → los datos no cumplen el esquema (campo faltante, valor fuera de rango, categoría inválida…)
    400 → la solicitud es válida pero no se puede atender (lote de más de 100 clientes)
    503 → la API está viva pero el modelo no está cargado (falta entrenar)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Permite ejecutar este archivo directamente (python api/main.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import config
from api.modelo import ModeloNoDisponible, ServicioModelo
from api.schemas import ClienteEntrada, LoteEntrada, LoteSalida, PrediccionSalida

estado: dict = {"servicio": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta UNA vez al arrancar: carga el modelo en memoria."""
    try:
        estado["servicio"] = ServicioModelo()
        estado["error"] = None
    except ModeloNoDisponible as error:
        # La API arranca igual, pero /health avisa y /predict responde 503.
        estado["servicio"], estado["error"] = None, str(error)
    yield
    estado["servicio"] = None


app = FastAPI(
    title=config.TITULO,
    description=config.DESCRIPCION,
    version=config.API_VERSION,
    lifespan=lifespan,
)


def obtener_servicio() -> ServicioModelo:
    if estado["servicio"] is None:
        # 503 = "Service Unavailable": el servidor funciona, pero no puede atender ahora
        raise HTTPException(
            status_code=503, detail=estado["error"] or "Modelo no cargado"
        )
    return estado["servicio"]


@app.get("/health", tags=["Monitoreo"])
def health() -> dict:
    """Chequeo de salud: lo usan los balanceadores y los sistemas de monitoreo."""
    servicio = estado["servicio"]
    return {
        "status": "healthy" if servicio else "degraded",
        "api_version": config.API_VERSION,
        "modelo_cargado": servicio is not None,
        "version_modelo": servicio.version if servicio else None,
    }


@app.get("/model-info", tags=["Monitoreo"])
def model_info() -> dict:
    """Metadatos del modelo: qué es, cuándo se entrenó, con qué umbral decide y qué tan bueno es."""
    return obtener_servicio().metadata


@app.post("/predict", response_model=PrediccionSalida, tags=["Predicción"])
def predict(cliente: ClienteEntrada) -> PrediccionSalida:
    """Probabilidad de abandono de un cliente y si conviene incluirlo en la campaña.

    Si los datos no cumplen el esquema (campo faltante, Meses_Permanencia = 100,
    Tipo_Contrato = "Anual"...), FastAPI responde **422** antes de llegar a esta función.
    """
    servicio = obtener_servicio()
    resultado = servicio.predecir([cliente])[0]
    return PrediccionSalida(**resultado, version_modelo=servicio.version)


@app.post("/predict-batch", response_model=LoteSalida, tags=["Predicción"])
def predict_batch(lote: LoteEntrada) -> LoteSalida:
    """Riesgo de varios clientes en una sola llamada (por ejemplo, la lista mensual del CRM)."""
    servicio = obtener_servicio()
    if len(lote.clientes) > config.MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {config.MAX_BATCH} clientes por solicitud; llegaron {len(lote.clientes)}",
        )
    predicciones = servicio.predecir(lote.clientes)
    return LoteSalida(
        predicciones=predicciones,
        cantidad=len(predicciones),
        en_riesgo=sum(p["abandona"] for p in predicciones),
        version_modelo=servicio.version,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        app_dir=str(Path(__file__).resolve().parents[1]),
    )
