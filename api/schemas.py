"""
Esquemas de entrada y salida con Pydantic.

FastAPI usa estas clases para:
    1. Validar lo que llega: si falta un campo, viene "dos" en lugar de 2 o una
       categoría que no existe, responde 422 explicando qué está mal.
    2. Documentar: la página /docs se genera a partir de estas clases.
    3. Convertir el JSON en objetos de Python (y viceversa).

Los nombres de los campos son los mismos de las columnas en español del notebook,
y las categorías son las del dataset original (en inglés).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Binaria = Literal[0, 1]
SiNoSinTelefono = Literal["Yes", "No", "No phone service"]
SiNoSinInternet = Literal["Yes", "No", "No internet service"]

EJEMPLO_CLIENTE = {
    "Genero": 0,
    "Adulto_Mayor": 0,
    "Tiene_Pareja": 0,
    "Tiene_Dependientes": 0,
    "Meses_Permanencia": 2,
    "Servicio_Telefonico": 1,
    "Multiples_Lineas": "No",
    "Servicio_Internet": "Fiber optic",
    "Seguridad_En_Linea": "No",
    "Respaldo_En_Linea": "No",
    "Proteccion_Dispositivo": "No",
    "Soporte_Tecnico": "No",
    "Streaming_TV": "No",
    "Streaming_Peliculas": "No",
    "Tipo_Contrato": "Month-to-month",
    "Facturacion_Electronica": 1,
    "Metodo_Pago": "Electronic check",
    "Cargo_Mensual": 70.7,
    "Cargo_Total": 151.65,
}


class ClienteEntrada(BaseModel):
    """Datos de un cliente. Los nombres coinciden con las columnas del notebook."""

    model_config = ConfigDict(json_schema_extra={"examples": [EJEMPLO_CLIENTE]})

    Genero: Binaria = Field(description="0 = mujer, 1 = hombre")
    Adulto_Mayor: Binaria = Field(description="1 si el cliente tiene 65 años o más")
    Tiene_Pareja: Binaria = Field(description="1 si tiene pareja")
    Tiene_Dependientes: Binaria = Field(description="1 si tiene personas a cargo")
    Meses_Permanencia: int = Field(
        ge=0, le=72, description="Meses como cliente (0 a 72)"
    )
    Servicio_Telefonico: Binaria = Field(description="1 si tiene servicio telefónico")
    Multiples_Lineas: SiNoSinTelefono
    Servicio_Internet: Literal["DSL", "Fiber optic", "No"]
    Seguridad_En_Linea: SiNoSinInternet
    Respaldo_En_Linea: SiNoSinInternet
    Proteccion_Dispositivo: SiNoSinInternet
    Soporte_Tecnico: SiNoSinInternet
    Streaming_TV: SiNoSinInternet
    Streaming_Peliculas: SiNoSinInternet
    Tipo_Contrato: Literal["Month-to-month", "One year", "Two year"]
    Facturacion_Electronica: Binaria = Field(
        description="1 si recibe la factura electrónica"
    )
    Metodo_Pago: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    Cargo_Mensual: float = Field(gt=0, le=500, description="Cargo mensual actual")
    Cargo_Total: float | None = Field(
        default=None,
        ge=0,
        description="Total facturado hasta hoy. Opcional: solo se usa para validar coherencia "
        "(el modelo no lo usa porque repite Meses x Cargo_Mensual)",
    )

    @model_validator(mode="after")
    def validar_coherencia(self):
        """Reglas que involucran VARIOS campos (FastAPI las devuelve como 422)."""
        if (
            self.Cargo_Total is not None
            and self.Meses_Permanencia > 1
            and self.Cargo_Total < self.Cargo_Mensual
        ):
            raise ValueError(
                "Cargo_Total no puede ser menor que Cargo_Mensual si el cliente lleva más de 1 mes"
            )
        if (
            self.Servicio_Telefonico == 0
            and self.Multiples_Lineas != "No phone service"
        ):
            raise ValueError(
                "Sin servicio telefónico, Multiples_Lineas debe ser 'No phone service'"
            )
        return self


class PrediccionSalida(BaseModel):
    """Respuesta de /predict."""

    probabilidad_abandono: float = Field(
        description="Probabilidad estimada de abandono (0 a 1)"
    )
    abandona: bool = Field(
        description="True si la probabilidad supera el umbral del modelo"
    )
    nivel_riesgo: Literal["bajo", "medio", "alto"] = Field(
        description="bajo: bajo el umbral · medio: entre el umbral y 0,70 · alto: 0,70 o más"
    )
    version_modelo: str


class LoteEntrada(BaseModel):
    """Entrada de /predict-batch: una lista de clientes (máximo 100)."""

    clientes: list[ClienteEntrada] = Field(min_length=1)


class PrediccionLote(BaseModel):
    probabilidad_abandono: float
    abandona: bool
    nivel_riesgo: Literal["bajo", "medio", "alto"]


class LoteSalida(BaseModel):
    predicciones: list[PrediccionLote]
    cantidad: int
    en_riesgo: int = Field(
        description="Cuántos clientes del lote quedaron marcados como 'abandona'"
    )
    version_modelo: str
