# Proyecto Equipo 3

## Integrantes
- Juan Pablo Madrigal Navarro
- Johana Marcela Hidalgo


## Cómo ejecutar
uv sync
uv run jupyter lab

## Respuestas del Módulo 00

### Paso 2: Revisar lo que se generó

**1. ¿Qué versión exacta de `pandas` quedó en `uv.lock`? ¿Qué rango quedó en `pyproject.toml`?**

- En `uv.lock` quedó la versión exacta **3.1.0**.
- En `pyproject.toml` quedó el rango **`pandas>=3`**.

`pyproject.toml` dice qué versiones se aceptan; `uv.lock` fija la versión que de verdad se instaló.

**2. ¿Cuántos paquetes hay en `uv.lock`? ¿Por qué son muchos más que los que instalaron?**

Hay **120 entradas `[[package]]`**, aunque solo instalamos 9 paquetes. Son muchas más porque cada paquete trae sus propias dependencias (dependencias transitivas). Por ejemplo, pandas trae `python-dateutil` y `tzdata`, y jupyter trae decenas (`tornado`, `jinja2`, `pyzmq`…). El lock las guarda todas para que el entorno sea idéntico en cualquier máquina.

**3. ¿Qué pasa si borran `.venv/` y ejecutan `uv run python -c "import pandas"`?**

No pasa nada malo: `uv run` detecta que falta el entorno, lo recrea automáticamente a partir de `uv.lock` (instaló 118 paquetes en menos de un segundo gracias al caché) y el comando funciona.

### Paso 3: Probar la reproducibilidad

**¿A los tres les salieron las mismas versiones?**

Sí: **pandas 3.1.0 y scikit-learn 1.9.1**. `uv sync` instala exactamente lo que dice `uv.lock`, no "lo último disponible".

### Reto opcional

**1. Script `src/hola.py`**

Imprime la versión de Python y de pandas. Al ejecutar `uv run python src/hola.py` muestra:

```
Python 3.14.2
pandas 3.1.0
```

**2. ¿Qué cambió en `uv.lock` con `uv add "pandas<3"`? ¿Y al volver a `uv add "pandas>=3"`?**

- Con `pandas<3`, uv bajó pandas de 3.0.6 a **2.3.3** y agregó una dependencia nueva, **`pytz`**, que pandas 2 necesita y pandas 3 no. El lock pasó de 120 a 121 paquetes y `pyproject.toml` cambió a `pandas<3`.
- Al volver a `pandas>=3`, uv instaló la versión más reciente que cumple el rango, **3.1.0** (no la 3.0.6 original), y quitó `pytz`; el lock volvió a 120 paquetes.

Conclusión: el rango de `pyproject.toml` solo pone límites; la versión exacta la decide uv al resolver y queda fijada en `uv.lock`.

**3. ¿Qué hace `uv export --format requirements-txt`? ¿Cuándo sería útil?**

Genera un `requirements.txt` con todas las versiones fijadas a partir del lock. Es útil cuando alguien o alguna herramienta no usa `uv`, por ejemplo un servidor, un Dockerfile, Google Colab o un compañero que trabaja con `pip`.

### Preguntas para la plenaria

**1. ¿Por qué `.venv/` no se comparte pero `uv.lock` sí?**

`.venv/` es pesado, depende del sistema operativo y del procesador de cada uno, y se puede regenerar. `uv.lock` es un archivo de texto pequeño con la "receta" exacta para regenerar el mismo entorno en cualquier máquina.

**2. ¿Qué error les salió y cómo lo resolvieron?**

El proyecto se había creado con Python 3.12 en lugar de 3.14. Lo corregimos con `uv python pin 3.14` y actualizando `requires-python` en `pyproject.toml`.

**3. En una frase: ¿qué ventaja tiene `uv` frente a `pip install` global?**

Cada proyecto tiene su propio entorno aislado con versiones fijadas en un lock, así que no hay conflictos entre proyectos y todos los integrantes tienen exactamente el mismo entorno, mucho más rápido.
