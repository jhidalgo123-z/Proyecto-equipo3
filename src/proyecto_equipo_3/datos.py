from pathlib import Path

import pandas as pd

DATASET = "blastchar/telco-customer-churn"
ARCHIVO = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / ARCHIVO


def cargar_datos(forzar_descarga=False):
    """Lee la copia de data/raw/; si no existe (o se pide), la descarga de Kaggle y la guarda."""
    if RUTA_RAW.exists() and not forzar_descarga:
        return pd.read_csv(RUTA_RAW)

    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET, ARCHIVO)
    RUTA_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_RAW, index=False)
    return df


if __name__ == "__main__":
    df = cargar_datos(forzar_descarga=True)
    print(f"Filas y columnas: {df.shape}")
    print("Primeros 5 registros:")
    print(df.head())
    print(f"Guardado en: {RUTA_RAW}")
