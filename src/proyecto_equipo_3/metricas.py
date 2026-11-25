"""
Métricas de clasificación reutilizables.

Regla de oro: un número solo no dice nada. Siempre se compara contra el
baseline, en train Y en test, y con varias métricas.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

# F2 = media entre recall y precision que le da el DOBLE de peso al recall:
# perder a un cliente (FN) nos cuesta más que una falsa alarma (FP).
# zero_division=0: el baseline nunca predice "abandona" y la precisión sería 0/0.
F2 = make_scorer(fbeta_score, beta=2, zero_division=0)

SCORING_CV = {
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f2": F2,
    "roc_auc": "roc_auc",
}

COLOR = "#00A86B"


def metricas_clasificacion(
    y_real, probabilidad, umbral: float = 0.5
) -> dict[str, float]:
    """Recall, precision, F1, F2, ROC-AUC y accuracy para un umbral dado."""
    y_pred = (np.asarray(probabilidad) >= umbral).astype(int)
    return {
        "recall": round(float(recall_score(y_real, y_pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_real, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_real, y_pred, zero_division=0)), 3),
        "f2": round(float(fbeta_score(y_real, y_pred, beta=2, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_real, probabilidad)), 3),
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 3),
    }


def tabla_train_test(
    modelos: dict, X_train, y_train, X_test, y_test, umbral=0.5
) -> pd.DataFrame:
    """Métricas en train y en test de cada modelo ya entrenado, con la brecha de recall."""
    filas = []
    for nombre, modelo in modelos.items():
        train = metricas_clasificacion(
            y_train, modelo.predict_proba(X_train)[:, 1], umbral
        )
        test = metricas_clasificacion(
            y_test, modelo.predict_proba(X_test)[:, 1], umbral
        )
        fila = {"modelo": nombre}
        for metrica in ["recall", "precision", "f1", "roc_auc", "accuracy"]:
            fila[f"{metrica}_train"] = train[metrica]
            fila[f"{metrica}_test"] = test[metrica]
        fila["brecha_recall"] = round(train["recall"] - test["recall"], 3)
        fila["brecha_auc"] = round(train["roc_auc"] - test["roc_auc"], 3)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def resumen_cv(resultados_cv: dict) -> pd.DataFrame:
    """Convierte {modelo: salida de cross_validate} en una tabla de medias ± desviación."""
    filas = {}
    for nombre, res in resultados_cv.items():
        fila = {}
        for metrica in SCORING_CV:
            valores = res[f"test_{metrica}"]
            fila[f"{metrica}_cv"] = round(float(np.mean(valores)), 3)
            fila[f"{metrica}_std"] = round(float(np.std(valores)), 3)
        filas[nombre] = fila
    return pd.DataFrame(filas).T.sort_values("f2_cv", ascending=False)


def tabla_umbrales(y_real, probabilidad, umbrales=None) -> pd.DataFrame:
    """Cómo cambian recall, precision, F1 y los errores al mover el umbral."""
    if umbrales is None:
        umbrales = np.round(np.arange(0.1, 0.91, 0.05), 2)
    y_real = np.asarray(y_real)
    filas = []
    for u in umbrales:
        y_pred = (np.asarray(probabilidad) >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
        filas.append(
            {
                "umbral": u,
                "recall": round(recall_score(y_real, y_pred, zero_division=0), 3),
                "precision": round(precision_score(y_real, y_pred, zero_division=0), 3),
                "f1": round(f1_score(y_real, y_pred, zero_division=0), 3),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "clientes_contactados": tp + fp,
            }
        )
    return pd.DataFrame(filas).set_index("umbral")


def elegir_umbral(tabla: pd.DataFrame, precision_min: float):
    """El umbral con MAYOR recall entre los que respetan la precisión mínima.

    El recall es la métrica principal y la precisión es la restricción de costo
    (no gastar la campaña en demasiadas falsas alarmas). Como el recall baja al
    subir el umbral, es el umbral más bajo que todavía cumple la precisión.
    Devuelve None si ningún umbral la cumple.
    """
    cumplen = tabla[tabla["precision"] >= precision_min]
    return float(cumplen["recall"].idxmax()) if not cumplen.empty else None


def grafico_umbrales(
    tabla: pd.DataFrame, umbral_elegido=None, recall_min=None, precision_min=None
):
    """Recall, precision y F1 según el umbral, con las metas del KPI."""
    _, ax = plt.subplots(1, 2, figsize=(14, 5))
    tabla[["recall", "precision", "f1"]].plot(ax=ax[0], marker="o")
    if recall_min is not None:
        ax[0].axhline(
            recall_min, ls="--", color="gray", lw=1, label=f"meta recall {recall_min}"
        )
    if precision_min is not None:
        ax[0].axhline(
            precision_min,
            ls=":",
            color="gray",
            lw=1,
            label=f"meta precision {precision_min}",
        )
    tabla[["FN", "FP"]].plot(ax=ax[1], marker="o", color=["#C2262B", "#F4A261"])
    for eje in ax:
        if umbral_elegido is not None:
            eje.axvline(
                umbral_elegido,
                color=COLOR,
                lw=2,
                label=f"umbral elegido {umbral_elegido}",
            )
        eje.set_xlabel("Umbral de decisión")
        eje.legend()
    ax[0].set_title("Métricas según el umbral")
    ax[1].set_title("Clientes mal clasificados según el umbral")
    ax[1].set_ylabel("Número de clientes")
    plt.tight_layout()
    plt.show()


def matriz_confusion(y_real, y_pred, titulo="Matriz de confusión"):
    """Grafica la matriz con etiquetas de negocio y devuelve (TN, FP, FN, TP)."""
    _, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_pred,
        display_labels=["Se queda (0)", "Abandona (1)"],
        cmap="Greens",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Lo que realmente pasó")
    plt.tight_layout()
    plt.show()
    return tuple(
        int(v) for v in confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    )


def curva_roc(modelos: dict, X, y):
    """Curva ROC de varios modelos sobre el mismo conjunto."""
    _, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo in modelos.items():
        RocCurveDisplay.from_estimator(modelo, X, y, name=nombre, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Azar (AUC = 0,5)")
    ax.set_title("Curva ROC en test")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def errores_por_segmento(
    df_test: pd.DataFrame, y_real, y_pred, columna: str
) -> pd.DataFrame:
    """Recall, precision y errores del modelo dentro de cada grupo de una variable."""
    datos = pd.DataFrame(
        {
            "grupo": df_test[columna].to_numpy(),
            "real": np.asarray(y_real),
            "pred": np.asarray(y_pred),
        }
    )
    filas = []
    for grupo, d in datos.groupby("grupo"):
        _, fp, fn, tp = confusion_matrix(d["real"], d["pred"], labels=[0, 1]).ravel()
        filas.append(
            {
                columna: grupo,
                "clientes": len(d),
                "tasa_abandono_real": round(d["real"].mean(), 3),
                "recall": round(tp / (tp + fn), 3) if tp + fn else np.nan,
                "precision": round(tp / (tp + fp), 3) if tp + fp else np.nan,
                "FN": fn,
                "FP": fp,
                "tasa_error": round((fp + fn) / len(d), 3),
            }
        )
    return pd.DataFrame(filas).set_index(columna)
