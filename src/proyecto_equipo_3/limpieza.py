import pandas as pd

COLUMNAS_ES = {
    "customerID": "ID_Cliente",
    "gender": "Genero",
    "SeniorCitizen": "Adulto_Mayor",
    "Partner": "Tiene_Pareja",
    "Dependents": "Tiene_Dependientes",
    "tenure": "Meses_Permanencia",
    "PhoneService": "Servicio_Telefonico",
    "MultipleLines": "Multiples_Lineas",
    "InternetService": "Servicio_Internet",
    "OnlineSecurity": "Seguridad_En_Linea",
    "OnlineBackup": "Respaldo_En_Linea",
    "DeviceProtection": "Proteccion_Dispositivo",
    "TechSupport": "Soporte_Tecnico",
    "StreamingTV": "Streaming_TV",
    "StreamingMovies": "Streaming_Peliculas",
    "Contract": "Tipo_Contrato",
    "PaperlessBilling": "Facturacion_Electronica",
    "PaymentMethod": "Metodo_Pago",
    "MonthlyCharges": "Cargo_Mensual",
    "TotalCharges": "Cargo_Total",
    "Churn": "Abandono_Cliente",
}

COLUMNAS_BINARIAS = [
    "Tiene_Pareja",
    "Tiene_Dependientes",
    "Servicio_Telefonico",
    "Facturacion_Electronica",
    "Abandono_Cliente",
]


def renombrar_columnas(df):
    """Traduce los nombres de las columnas al español."""
    return df.rename(columns=COLUMNAS_ES)


def codificar_binarias(df):
    """Codifica Genero (Female=0, Male=1) y las variables Yes/No como 1/0."""
    df = df.copy()
    df["Genero"] = df["Genero"].map({"Female": 0, "Male": 1})
    for col in COLUMNAS_BINARIAS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def convertir_cargo_total(df):
    """Convierte Cargo_Total a numérico; los valores en blanco quedan como NaN."""
    df = df.copy()
    df["Cargo_Total"] = pd.to_numeric(
        df["Cargo_Total"].astype(str).str.strip(),
        errors="coerce",
    )
    return df


def Limpieza(df):
    """Aplica todo el proceso de limpieza sobre el dataset original de Kaggle."""
    df = renombrar_columnas(df)
    df = codificar_binarias(df)
    df = convertir_cargo_total(df)
    return df
