"""
predecir.py: usa el modelo guardado para estimar el riesgo de abandono.

Uso (desde la raíz del repo)::

    # Un cliente, en JSON (texto o ruta a un archivo .json)
    uv run python scripts/predecir.py --json ejemplos/cliente.json

    # Muchos clientes (predicción por lotes) desde un CSV con las columnas en español
    uv run python scripts/predecir.py --csv data/processed/clientes.csv --salida reports/predicciones.csv

Requiere haber corrido antes ``scripts/train.py``.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proyecto_equipo_3 import modelo as md


def leer_json(texto: str) -> pd.DataFrame:
    """Acepta un JSON en texto o la ruta a un archivo .json (un cliente o una lista)."""
    ruta = Path(texto)
    contenido = (
        ruta.read_text(encoding="utf-8")
        if ruta.suffix == ".json" and ruta.exists()
        else texto
    )
    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as error:
        sys.exit(f"[ERROR] El JSON no es válido: {error}")
    if (
        isinstance(datos, dict) and "clientes" in datos
    ):  # mismo formato que /predict-batch
        datos = datos["clientes"]
    return pd.DataFrame(datos if isinstance(datos, list) else [datos])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice el riesgo de abandono con el modelo guardado"
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--json", help="Un cliente en JSON (texto o ruta a archivo .json)"
    )
    grupo.add_argument(
        "--csv", type=Path, help="CSV con varios clientes (columnas en español)"
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("reports/predicciones.csv"),
        help="Dónde guardar las predicciones (por defecto reports/predicciones.csv)",
    )
    args = parser.parse_args()

    if not md.RUTA_MODELO.exists():
        sys.exit(
            "[ERROR] No hay modelo entrenado. Primero ejecuta: uv run python scripts/train.py"
        )
    modelo = joblib.load(md.RUTA_MODELO)
    metadata = (
        json.loads(md.RUTA_METADATA.read_text(encoding="utf-8"))
        if md.RUTA_METADATA.exists()
        else {}
    )
    umbral = float(metadata.get("umbral", 0.5))

    if args.json:
        datos = leer_json(args.json)
    else:
        if not args.csv.exists():
            sys.exit(f"[ERROR] No existe el archivo {args.csv}")
        try:
            datos = pd.read_csv(args.csv)
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            sys.exit(f"[ERROR] {args.csv} no es un CSV válido: {error}")

    faltan = [c for c in md.FEATURES if c not in datos.columns]
    if faltan:
        sys.exit(f"[ERROR] Faltan {len(faltan)} columnas: {faltan}")

    probabilidad = modelo.predict_proba(datos[md.FEATURES])[:, 1]
    resultado = pd.DataFrame(
        {
            "ID_Cliente": datos.get("ID_Cliente", datos.index),
            "probabilidad_abandono": probabilidad.round(3),
            "abandona": (probabilidad >= umbral).astype(int),
        }
    )

    print(f"Modelo versión {metadata.get('version', '?')} · umbral {umbral}")
    if args.json and len(resultado) == 1:
        fila = resultado.iloc[0]
        print(f"Probabilidad de abandono: {fila['probabilidad_abandono']:.1%}")
        print(
            f"Clase predicha: {'ABANDONA (1)' if fila['abandona'] else 'SE QUEDA (0)'}"
        )
        return

    print(resultado.head(10).to_string(index=False))
    print(
        f"… {len(resultado)} clientes · {resultado['abandona'].sum()} marcados en riesgo "
        f"({resultado['abandona'].mean():.1%})"
    )
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False)
    print(f"Guardado en {args.salida}")


if __name__ == "__main__":
    main()
