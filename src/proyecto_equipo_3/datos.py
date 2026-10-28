from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "blastchar/telco-customer-churn"
ARCHIVO = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / ARCHIVO


def cargar_datos():
    """Descarga el dataset de Kaggle y guarda una copia en data/raw/."""
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET, ARCHIVO)
    df.to_csv(RUTA_RAW, index=False)
    return df


if __name__ == "__main__":
    df = cargar_datos()
    print(f"Filas y columnas: {df.shape}")
    print("Primeros 5 registros:")
    print(df.head())
    print(f"Guardado en: {RUTA_RAW}")
