"""
Construcción del pipeline de modelado (preprocesamiento + modelo).

Todo vive en UN solo objeto de scikit-learn: así el mismo preprocesamiento
(imputar, escalar, one-hot) se aplica igual en entrenamiento, en el notebook,
en predecir.py y en la API. Esa es la forma de evitar el data leakage y los
errores de "en mi notebook funcionaba".
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42
OBJETIVO = "Abandono_Cliente"
COLUMNAS_EXCLUIDAS = ["ID_Cliente"]  # identificador: no aporta para predecir

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MODELOS = RAIZ / "models"
RUTA_MODELO = CARPETA_MODELOS / "churn_pipeline.joblib"
RUTA_METADATA = CARPETA_MODELOS / "metadata.json"

# Cargo_Total se excluye: es casi exactamente Meses_Permanencia x Cargo_Mensual
# (correlación 0,9996), así que repite información (ver sección 9.1 del notebook).
VARIABLES_NUMERICAS = ["Meses_Permanencia", "Cargo_Mensual"]

# Variables ya codificadas como 0/1 en limpieza.py
VARIABLES_BINARIAS = [
    "Genero",
    "Adulto_Mayor",
    "Tiene_Pareja",
    "Tiene_Dependientes",
    "Servicio_Telefonico",
    "Facturacion_Electronica",
]

VARIABLES_CATEGORICAS = [
    "Multiples_Lineas",
    "Servicio_Internet",
    "Seguridad_En_Linea",
    "Respaldo_En_Linea",
    "Proteccion_Dispositivo",
    "Soporte_Tecnico",
    "Streaming_TV",
    "Streaming_Peliculas",
    "Tipo_Contrato",
    "Metodo_Pago",
]

FEATURES = VARIABLES_NUMERICAS + VARIABLES_BINARIAS + VARIABLES_CATEGORICAS


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve las variables del modelo (X) y el objetivo (y)."""
    return df[FEATURES], df[OBJETIVO]


def construir_preprocesador(numericas=VARIABLES_NUMERICAS) -> ColumnTransformer:
    """Numéricas: mediana + escalado. Categóricas y binarias: moda + one-hot."""
    pasos_numericos = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    pasos_categoricos = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": una categoría nueva en producción no rompe el modelo
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numericas", pasos_numericos, list(numericas)),
            (
                "categoricas",
                pasos_categoricos,
                VARIABLES_BINARIAS + VARIABLES_CATEGORICAS,
            ),
        ]
    )


def construir_pipeline(modelo, numericas=VARIABLES_NUMERICAS) -> Pipeline:
    """Une el preprocesamiento con un modelo en un solo objeto."""
    return Pipeline(
        [
            ("preprocesamiento", construir_preprocesador(numericas)),
            ("modelo", modelo),
        ]
    )


def modelos_candidatos() -> dict:
    """Modelos de complejidad creciente: del baseline a los ensembles."""
    return {
        "Baseline (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMILLA
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=SEMILLA
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=SEMILLA, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEMILLA
        ),
    }


# Criterio de éxito (KPI del modelo), definido ANTES de modelar. Se mide en test.
CRITERIO_EXITO = {"recall": 0.75, "precision": 0.50, "roc_auc": 0.80}

# Modelo final elegido en el notebook (sección 9): regresión logística con los
# hiperparámetros que encontró GridSearchCV (scoring F2, 5 folds).
NOMBRE_MODELO_FINAL = "Regresión logística"
HIPERPARAMETROS_FINALES = {"C": 0.01, "solver": "liblinear"}


def construir_modelo_final(**hiperparametros) -> Pipeline:
    """Pipeline del modelo final; por defecto con los hiperparámetros elegidos."""
    parametros = {**HIPERPARAMETROS_FINALES, **hiperparametros}
    return construir_pipeline(
        LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMILLA, **parametros
        )
    )


def guardar_artefacto(pipeline: Pipeline, metadata: dict) -> None:
    """Guarda el pipeline (joblib) y sus metadatos (JSON) en models/."""
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, RUTA_MODELO)
    RUTA_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def cargar_artefacto() -> tuple[Pipeline, dict]:
    """Carga el pipeline y sus metadatos; avisa claramente si no se ha entrenado."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_MODELO}. Entrena primero con: uv run python scripts/train.py"
        )
    metadata = (
        json.loads(RUTA_METADATA.read_text(encoding="utf-8"))
        if RUTA_METADATA.exists()
        else {}
    )
    return joblib.load(RUTA_MODELO), metadata


# Grillas para GridSearchCV. El prefijo "modelo__" apunta al paso del pipeline.
GRILLAS = {
    "Regresión logística": {
        "modelo__C": [0.01, 0.1, 1, 10],
        "modelo__solver": ["lbfgs", "liblinear"],
    },
    "Árbol de decisión": {
        "modelo__max_depth": [3, 4, 5, 6, 8],
        "modelo__min_samples_leaf": [1, 20, 50],
    },
    "Random Forest": {
        "modelo__max_depth": [5, 8, 12, None],
        "modelo__min_samples_leaf": [1, 10, 30],
    },
    "Gradient Boosting": {
        "modelo__learning_rate": [0.03, 0.1],
        "modelo__max_depth": [3, None],
        "modelo__min_samples_leaf": [20, 50],
    },
}
