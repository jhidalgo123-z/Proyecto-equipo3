"""
train.py: entrena el pipeline de abandono de clientes y lo guarda listo para producción.

Uso (desde la raíz del repo)::

    uv run python scripts/train.py                    # hiperparámetros ya elegidos en el notebook
    uv run python scripts/train.py --buscar           # vuelve a buscarlos con GridSearchCV (más lento)
    uv run python scripts/train.py --version 1.1.0

Qué hace, paso a paso:
    1. Carga data/raw/ y aplica la limpieza (limpieza.py).
    2. Divide train/test (80/20), estratificado y con semilla 42.
    3. Construye el pipeline (preprocesamiento + regresión logística).
    4. (Opcional) Busca hiperparámetros con GridSearchCV (5 folds, scoring F2).
    5. Elige el umbral con predicciones de validación cruzada sobre TRAIN
       (no se mira el test para elegir nada).
    6. Evalúa en train y en test y verifica el criterio de éxito del KPI.
    7. Reentrena con TODOS los datos y guarda:
         models/churn_pipeline.joblib → el pipeline completo
         models/metadata.json         → versión, fecha, variables, hiperparámetros, métricas y umbral

¿Por qué reentrenar con todo al final?
    El test ya cumplió su función: estimar cómo le irá al modelo con clientes nuevos.
    Para producción queremos que aprenda de la mayor cantidad de datos posible.
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

# Permite importar el paquete aunque no esté instalado (mismo truco que el notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proyecto_equipo_3 import metricas as mt
from proyecto_equipo_3 import modelo as md
from proyecto_equipo_3.datos import cargar_datos
from proyecto_equipo_3.limpieza import Limpieza

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y guarda el pipeline de abandono de clientes"
    )
    parser.add_argument(
        "--buscar", action="store_true", help="Buscar hiperparámetros con GridSearchCV"
    )
    # Versionado semántico MAYOR.MENOR.PARCHE:
    #   MAYOR → cambia lo que recibe o devuelve el modelo (p. ej., otras variables): rompe a quien lo usa
    #   MENOR → mejora compatible (reentreno con más datos, otros hiperparámetros)
    #   PARCHE → corrección pequeña que no cambia el comportamiento esperado
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Versión del modelo (semver, por defecto 1.0.0)",
    )
    args = parser.parse_args()

    # 1-2. Datos
    try:
        df = Limpieza(cargar_datos())
    except Exception as error:  # noqa: BLE001  (sin archivo local y sin conexión a Kaggle)
        sys.exit(f"[ERROR] No se pudieron cargar los datos: {error}")
    X, y = md.separar_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=md.SEMILLA
    )
    print(
        f"Datos: {len(df)} clientes · train={len(X_train)} · test={len(X_test)} · abandono={y.mean():.1%}"
    )

    # 3-4. Pipeline (+ búsqueda opcional)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=md.SEMILLA)
    hiperparametros = dict(md.HIPERPARAMETROS_FINALES)
    if args.buscar:
        print("Buscando hiperparámetros (GridSearchCV, 5 folds, scoring F2)...")
        busqueda = GridSearchCV(
            md.construir_modelo_final(),
            md.GRILLAS[md.NOMBRE_MODELO_FINAL],
            cv=cv,
            scoring=mt.SCORING_CV,
            refit="f2",
            n_jobs=-1,
        )
        busqueda.fit(X_train, y_train)
        hiperparametros = {
            k.split("__")[-1]: v for k, v in busqueda.best_params_.items()
        }
        print(f"   Mejores: {hiperparametros} · F2 CV = {busqueda.best_score_:.3f}")
    pipeline = md.construir_modelo_final(**hiperparametros)

    # 5. Umbral elegido con validación cruzada (out-of-fold) sobre train
    prob_oof = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    umbral = mt.elegir_umbral(
        mt.tabla_umbrales(y_train, prob_oof), md.CRITERIO_EXITO["precision"]
    )
    if umbral is None:
        umbral = 0.5
        print(
            "[AVISO] Ningún umbral logra la precisión mínima en validación; se usa 0,5."
        )
    print(f"Umbral elegido (validación cruzada en train): {umbral}")

    # 6. Evaluación en train y en test
    pipeline.fit(X_train, y_train)
    metricas_train = mt.metricas_clasificacion(
        y_train, pipeline.predict_proba(X_train)[:, 1], umbral
    )
    metricas_test = mt.metricas_clasificacion(
        y_test, pipeline.predict_proba(X_test)[:, 1], umbral
    )
    print(f"Train: {metricas_train}")
    print(f"Test:  {metricas_test}")
    detalle = {m: metricas_test[m] >= minimo for m, minimo in md.CRITERIO_EXITO.items()}
    cumple = all(detalle.values())
    for m, ok in detalle.items():
        print(
            f"   {m:<9} test = {metricas_test[m]:.3f}  (meta >= {md.CRITERIO_EXITO[m]})  {'OK' if ok else 'NO'}"
        )
    print(f"   Criterio de éxito del KPI: {'CUMPLE' if cumple else 'NO CUMPLE'}")

    # 7. Reentrenar con todos los datos y guardar
    pipeline.fit(X, y)
    metadata = {
        "nombre": "churn_pipeline",
        "modelo": "LogisticRegression(class_weight='balanced') dentro de un Pipeline de scikit-learn",
        "version": args.version,
        "entrenado_en": datetime.now().isoformat(timespec="seconds"),
        "semilla": md.SEMILLA,
        "objetivo": md.OBJETIVO,
        "features_numericas": md.VARIABLES_NUMERICAS,
        "features_binarias": md.VARIABLES_BINARIAS,
        "features_categoricas": md.VARIABLES_CATEGORICAS,
        "hiperparametros": hiperparametros,
        "umbral": umbral,
        "criterio_exito": md.CRITERIO_EXITO,
        "metricas_train": metricas_train,
        "metricas_test": metricas_test,
        "cumple_criterio_exito": cumple,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_entrenamiento_final": len(X),
    }
    md.guardar_artefacto(pipeline, metadata)
    print(f"Modelo:    {md.RUTA_MODELO.relative_to(md.RAIZ)}")
    print(f"Metadatos: {md.RUTA_METADATA.relative_to(md.RAIZ)}")


if __name__ == "__main__":
    main()
