# Riesgo de abandono de clientes (Telco Customer Churn) · Proyecto Equipo 3

Proyecto final del curso **Proyecto I: Introducción a la Analítica**, Especialización en Ciencia de Datos e Inteligencia Artificial, Universidad de Medellín.

## Equipo

- Juan Pablo Madrigal Navarro
- Johana Marcela Hidalgo Zuluaga

## El problema

Una empresa de telecomunicaciones pierde al **26,5 %** de sus clientes. La gerencia de retención necesita decidir **a qué clientes ofrecer una campaña de retención** antes de que cancelen. Construimos un modelo de clasificación que estima la probabilidad de abandono de cada cliente.

| KPI | Meta |
|---|---|
| **De negocio** | Bajar la tasa de abandono del 26,5 % al 22 % en 6 meses con campañas dirigidas a los clientes que marque el modelo |
| **Del modelo (criterio de éxito)** | En test: recall de la clase "abandona" ≥ 0,75, precisión ≥ 0,50, ROC-AUC ≥ 0,80, y superar al baseline |

**Métrica principal: recall**, porque un cliente que se va sin ser detectado (FN) cuesta mucho más que un descuento ofrecido a quien se iba a quedar (FP). La accuracy no se usa: un modelo que nunca detecta a nadie ya obtiene un 73,5 %.

## Dataset

- **Telco Customer Churn** (IBM), publicado en Kaggle por BlastChar: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- 7.043 clientes (una fila = un cliente) y 21 columnas: datos demográficos, servicios contratados, contrato y facturación.
- Copia en `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (≈ 1 MB). Si no existe, `cargar_datos()` la descarga de Kaggle (`uv run python src/proyecto_equipo_3/datos.py`).

## Cómo ejecutar

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.14.

```bash
# 1. Entorno reproducible (versiones fijadas en uv.lock)
uv sync

# 2. Notebook completo: carga, limpieza, EDA, modelo y conclusiones
uv run jupyter lab notebooks/CustomerChurn.ipynb        # luego "Restart Kernel and Run All"

# 3. Entrenar y guardar el modelo (models/churn_pipeline.joblib + models/metadata.json)
uv run python scripts/train.py                  # hiperparámetros ya elegidos
uv run python scripts/train.py --buscar --version 1.1.0   # vuelve a buscarlos con GridSearchCV

# 4. Predecir con el modelo guardado
uv run python scripts/predecir.py --json ejemplos/cliente.json
uv run python scripts/predecir.py --csv mis_clientes.csv --salida reports/predicciones.csv

# 5. API (documentación interactiva en http://127.0.0.1:8000/docs)
uv run uvicorn api.main:app --reload

# 6. Pruebas
uv run pytest -v
```

### Probar la API con curl

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @ejemplos/cliente.json
curl -X POST http://127.0.0.1:8000/predict-batch -H "Content-Type: application/json" -d @ejemplos/clientes.json
```

Respuesta de `/predict` para el cliente de ejemplo (2 meses, fibra óptica, contrato mes a mes y cheque electrónico):

```json
{"probabilidad_abandono": 0.8482, "abandona": true, "nivel_riesgo": "alto", "version_modelo": "1.0.0"}
```

| Endpoint | Qué hace | Errores |
|---|---|---|
| `GET /health` | Estado de la API y del modelo | |
| `GET /model-info` | Versión, variables, hiperparámetros, umbral y métricas de train y test | 503 sin modelo |
| `POST /predict` | Probabilidad, clase (`abandona`) y nivel de riesgo de un cliente | 422 datos inválidos · 503 sin modelo |
| `POST /predict-batch` | Lo mismo para hasta 100 clientes | 400 si hay más de 100 · 422 · 503 |

## Resultados

Partición 80/20 estratificada (semilla 42) y validación cruzada de 5 folds en train para comparar modelos, ajustar hiperparámetros (GridSearchCV, scoring F2) y elegir el umbral. El test se usó una sola vez.

| Modelo (test, umbral 0,5) | Recall | Precisión | ROC-AUC | Accuracy |
|---|---|---|---|---|
| Baseline (siempre "se queda") | 0,000 | — | 0,500 | 0,735 |
| Árbol de decisión (prof. 5) | 0,775 | 0,529 | 0,834 | 0,757 |
| Random Forest (sobreajustado: recall 1,00 en train) | 0,610 | 0,529 | 0,812 | 0,752 |
| Gradient Boosting | 0,754 | 0,525 | 0,833 | 0,754 |
| **Regresión logística final (C = 0,01)** | **0,789** | **0,510** | **0,839** | 0,743 |

- **Cumple el criterio de éxito** en las tres metas. Sin overfitting: recall de 0,799 en train y 0,789 en test.
- **En unidades de negocio:** de 374 clientes del test que se fueron, el modelo detectó **295** (el baseline, 0). De los 578 clientes marcados en riesgo, 295 sí se iban a ir.
- **Lo que más pesa:** meses de permanencia, contrato mes a mes, fibra óptica, cargo mensual y pago con cheque electrónico. El soporte técnico y la seguridad en línea reducen el riesgo.

## Límites

- **Casi no detecta el abandono en contratos de 1 y 2 años** (recall de 0,22 y 0) **ni en clientes sin internet**. Para esos grupos se necesitan otras señales (por ejemplo, la fecha de fin del contrato).
- La precisión está justo en el límite (0,51): cerca de la mitad de los contactos son falsas alarmas.
- Es una foto de un solo momento (sin fechas) de una empresa ficticia. Hay que validar con datos propios, monitorear el *data drift* y reentrenar.
- Asociación no es causalidad. La efectividad real de la campaña debe medirse con un piloto A/B.
- No hay sesgo por género. Los adultos mayores reciben más falsas alarmas.

## Estructura del repositorio

```
├── api/                    API con FastAPI (config, schemas, modelo, main)
├── data/raw/               Dataset original
├── ejemplos/               cliente.json y clientes.json para probar la API
├── models/                 Pipeline y metadata (se generan; no se suben)
├── notebooks/
│   └── CustomerChurn.ipynb Proyecto completo (negocio → EDA → modelo → conclusiones)
├── scripts/                train.py y predecir.py
├── src/proyecto_equipo_3/  datos, limpieza, análisis estadístico, modelo y métricas
└── tests/test_api.py       Pruebas de la API (pytest)
```

> **Nota para macOS:** si `uv run python -c "import proyecto_equipo_3"` falla con Python 3.14, es porque macOS marcó como oculto el archivo `.pth` de `.venv`. El notebook, los scripts, la API y pytest agregan `src/` al path, así que funcionan igual.

---

## Anexo: respuestas del Módulo 00

#### Paso 2: Revisar lo que se generó

**1. ¿Qué versión exacta de `pandas` quedó en `uv.lock`? ¿Qué rango quedó en `pyproject.toml`?**

- En `uv.lock` quedó la versión exacta **3.1.0**.
- En `pyproject.toml` quedó el rango **`pandas>=3`**.

`pyproject.toml` dice qué versiones se aceptan; `uv.lock` fija la versión que de verdad se instaló.

**2. ¿Cuántos paquetes hay en `uv.lock`? ¿Por qué son muchos más que los que instalaron?**

Hay **120 entradas `[[package]]`**, aunque solo instalamos 9 paquetes. Son muchas más porque cada paquete trae sus propias dependencias (dependencias transitivas). Por ejemplo, pandas trae `python-dateutil` y `tzdata`, y jupyter trae decenas (`tornado`, `jinja2`, `pyzmq`…). El lock las guarda todas para que el entorno sea idéntico en cualquier máquina.

**3. ¿Qué pasa si borran `.venv/` y ejecutan `uv run python -c "import pandas"`?**

No pasa nada malo: `uv run` detecta que falta el entorno, lo recrea automáticamente a partir de `uv.lock` (instaló 118 paquetes en menos de un segundo gracias al caché) y el comando funciona.

#### Paso 3: Probar la reproducibilidad

**¿A los tres les salieron las mismas versiones?**

Sí: **pandas 3.1.0 y scikit-learn 1.9.1**. `uv sync` instala exactamente lo que dice `uv.lock`, no "lo último disponible".

#### Reto opcional


**2. ¿Qué cambió en `uv.lock` con `uv add "pandas<3"`? ¿Y al volver a `uv add "pandas>=3"`?**

- Con `pandas<3`, uv bajó pandas de 3.0.6 a **2.3.3** y agregó una dependencia nueva, **`pytz`**, que pandas 2 necesita y pandas 3 no. El lock pasó de 120 a 121 paquetes y `pyproject.toml` cambió a `pandas<3`.
- Al volver a `pandas>=3`, uv instaló la versión más reciente que cumple el rango, **3.1.0** (no la 3.0.6 original), y quitó `pytz`; el lock volvió a 120 paquetes.

Conclusión: el rango de `pyproject.toml` solo pone límites; la versión exacta la decide uv al resolver y queda fijada en `uv.lock`.

**3. ¿Qué hace `uv export --format requirements-txt`? ¿Cuándo sería útil?**

Genera un `requirements.txt` con todas las versiones fijadas a partir del lock. Es útil cuando alguien o alguna herramienta no usa `uv`, por ejemplo un servidor, un Dockerfile, Google Colab o un compañero que trabaja con `pip`.

#### Preguntas para la plenaria

**1. ¿Por qué `.venv/` no se comparte pero `uv.lock` sí?**

`.venv/` es pesado, depende del sistema operativo y del procesador de cada uno, y se puede regenerar. `uv.lock` es un archivo de texto pequeño con la "receta" exacta para regenerar el mismo entorno en cualquier máquina.

**2. ¿Qué error les salió y cómo lo resolvieron?**

El proyecto se había creado con Python 3.12 en lugar de 3.14. Lo corregimos con `uv python pin 3.14` y actualizando `requires-python` en `pyproject.toml`.

**3. En una frase: ¿qué ventaja tiene `uv` frente a `pip install` global?**

Cada proyecto tiene su propio entorno aislado con versiones fijadas en un lock, así que no hay conflictos entre proyectos y todos los integrantes tienen exactamente el mismo entorno, mucho más rápido.
