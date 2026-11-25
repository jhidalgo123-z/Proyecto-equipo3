"""
Pruebas de la API con pytest y el TestClient de FastAPI.

Ejecutar (desde la raíz del repo)::

    uv run pytest -v

El TestClient simula solicitudes HTTP SIN levantar un servidor real.
Las pruebas de predicción necesitan el modelo entrenado (scripts/train.py);
si no existe, se saltan (skip) con un mensaje claro.
"""

import pytest
from fastapi.testclient import TestClient

from api import config
from api.main import app
from api.schemas import EJEMPLO_CLIENTE

requiere_modelo = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Primero entrena el modelo: uv run python scripts/train.py",
)


@pytest.fixture(scope="module")
def cliente():
    # El "with" ejecuta el lifespan (carga del modelo)
    with TestClient(app) as c:
        yield c


def test_health_responde(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["api_version"] == config.API_VERSION


@requiere_modelo
def test_health_modelo_cargado(cliente):
    assert cliente.get("/health").json()["modelo_cargado"] is True


@requiere_modelo
def test_model_info_cumple_criterio(cliente):
    info = cliente.get("/model-info").json()
    assert {"metricas_train", "metricas_test", "umbral", "version"} <= info.keys()
    for metrica, minimo in info["criterio_exito"].items():
        assert info["metricas_test"][metrica] >= minimo


@requiere_modelo
def test_predict_cliente_valido(cliente):
    respuesta = cliente.post("/predict", json=EJEMPLO_CLIENTE)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert 0 <= cuerpo["probabilidad_abandono"] <= 1
    assert cuerpo["nivel_riesgo"] in {"bajo", "medio", "alto"}
    assert isinstance(cuerpo["abandona"], bool)
    assert cuerpo["version_modelo"] == "1.0.0"


@requiere_modelo
def test_contrato_largo_menos_riesgo(cliente):
    """Prueba de sentido común: el mismo cliente con contrato de 2 años y 60 meses tiene menos riesgo."""
    nuevo = cliente.post("/predict", json=EJEMPLO_CLIENTE).json()[
        "probabilidad_abandono"
    ]
    leal = {
        **EJEMPLO_CLIENTE,
        "Tipo_Contrato": "Two year",
        "Meses_Permanencia": 60,
        "Cargo_Total": 4200.0,
    }
    assert cliente.post("/predict", json=leal).json()["probabilidad_abandono"] < nuevo


@pytest.mark.parametrize(
    "cambio",
    [
        {"Meses_Permanencia": 100},  # fuera de 0-72
        {"Meses_Permanencia": -1},  # negativo
        {"Cargo_Mensual": 0},  # debe ser > 0
        {"Tipo_Contrato": "Anual"},  # categoría no permitida
        {"Genero": 2},  # binaria fuera de {0, 1}
        {"Cargo_Mensual": "caro"},  # tipo incorrecto
    ],
)
def test_predict_datos_invalidos_422(cliente, cambio):
    respuesta = cliente.post("/predict", json={**EJEMPLO_CLIENTE, **cambio})
    assert respuesta.status_code == 422


def test_predict_campo_faltante_422(cliente):
    incompleto = {k: v for k, v in EJEMPLO_CLIENTE.items() if k != "Tipo_Contrato"}
    assert cliente.post("/predict", json=incompleto).status_code == 422


def test_cargo_total_incoherente_422(cliente):
    """Regla entre dos campos: con más de 1 mes, el total no puede ser menor que un solo mes."""
    incoherente = {**EJEMPLO_CLIENTE, "Meses_Permanencia": 12, "Cargo_Total": 10.0}
    assert cliente.post("/predict", json=incoherente).status_code == 422


@requiere_modelo
def test_batch(cliente):
    clientes = [EJEMPLO_CLIENTE, {**EJEMPLO_CLIENTE, "Tipo_Contrato": "Two year"}]
    respuesta = cliente.post("/predict-batch", json={"clientes": clientes})
    assert respuesta.status_code == 200
    assert respuesta.json()["cantidad"] == 2


@requiere_modelo
def test_batch_mas_de_100_400(cliente):
    respuesta = cliente.post(
        "/predict-batch", json={"clientes": [EJEMPLO_CLIENTE] * 101}
    )
    assert respuesta.status_code == 400


def test_batch_vacio_422(cliente):
    assert cliente.post("/predict-batch", json={"clientes": []}).status_code == 422


def test_sin_modelo_503(cliente, monkeypatch):
    """Si el modelo no se pudo cargar, la API sigue viva pero /predict responde 503."""
    from api import main

    monkeypatch.setitem(
        main.estado, "servicio", None
    )  # se restaura al terminar la prueba
    monkeypatch.setitem(main.estado, "error", "Modelo no cargado")
    assert cliente.post("/predict", json=EJEMPLO_CLIENTE).status_code == 503
    assert cliente.get("/health").json()["status"] == "degraded"
