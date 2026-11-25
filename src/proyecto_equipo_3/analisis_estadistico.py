import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

VARIABLES_NUMERICAS = [
    "Meses_Permanencia",
    "Cargo_Mensual",
    "Cargo_Total",
]

VARIABLES_CATEGORICAS = [
    "Genero",
    "Adulto_Mayor",
    "Tiene_Pareja",
    "Tiene_Dependientes",
    "Servicio_Telefonico",
    "Multiples_Lineas",
    "Servicio_Internet",
    "Tipo_Contrato",
    "Facturacion_Electronica",
    "Metodo_Pago",
    "Abandono_Cliente",
]

VARIABLES_SEMIBINARIAS = [
    "Multiples_Lineas",
    "Seguridad_En_Linea",
    "Respaldo_En_Linea",
    "Proteccion_Dispositivo",
    "Soporte_Tecnico",
    "Streaming_TV",
    "Streaming_Peliculas",
]

ETIQUETAS_GENERO = {0: "Mujer", 1: "Hombre"}
ETIQUETAS_BINARIAS = {0: "No", 1: "Sí"}


def _etiquetas(df, columna, valores):
    """Traduce los códigos 0/1 a etiquetas legibles para gráficos y tablas."""
    if columna == "Genero":
        return [ETIQUETAS_GENERO.get(v, v) for v in valores]
    if set(df[columna].dropna().unique()) <= {0, 1}:
        return [ETIQUETAS_BINARIAS.get(v, v) for v in valores]
    return list(valores)


def valores_unicos(df):
    """Imprime los valores únicos de cada columna (excepto el identificador)."""
    for col in df.columns.drop("ID_Cliente", errors="ignore"):
        print(f"{col}:\n {df[col].unique()}")


def conteo_semibinarias(df):
    """Muestra la distribución de las variables con categorías 'No ... service'."""
    for col in VARIABLES_SEMIBINARIAS:
        print(f"\n{col}")
        print(df[col].value_counts())


def grafico_distribucion_categoricas(df, columnas=VARIABLES_CATEGORICAS):
    """Gráficos de barras con frecuencia absoluta y porcentaje por categoría."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    axes = axes.flatten()

    for i, columna in enumerate(columnas):
        frecuencia = df[columna].value_counts().sort_index()
        frecuencia.index = _etiquetas(df, columna, frecuencia.index)
        frecuencia.plot(kind="bar", ax=axes[i], color="skyblue")

        axes[i].set_title(f"Distribución de {columna}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frecuencia")

        total = frecuencia.sum()
        for j, v in enumerate(frecuencia):
            axes[i].text(
                j,
                v + frecuencia.max() * 0.03,
                f"{v}\n({v / total * 100:.1f}%)",
                ha="center",
                fontsize=9,
            )

    for j in range(len(columnas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def boxplots_numericas(df):
    """Diagramas de cajas y bigotes de las variables numéricas."""
    sns.set_theme(style="whitegrid")
    _, axs = plt.subplots(1, 3, figsize=(18, 6))
    config = [
        ("Meses_Permanencia", "Meses de Permanencia", "Meses", "skyblue"),
        ("Cargo_Mensual", "Cargo Mensual", "Valor", "lightgreen"),
        ("Cargo_Total", "Cargo Total", "Valor", "salmon"),
    ]
    for ax, (columna, titulo, etiqueta_y, color) in zip(axs, config):
        sns.boxplot(y=df[columna], ax=ax, color=color)
        ax.set_title(titulo)
        ax.set_ylabel(etiqueta_y)

    plt.tight_layout()
    plt.show()


def resumen_por_abandono(df, variables=VARIABLES_NUMERICAS):
    """Estadísticos descriptivos de cada variable numérica según Abandono_Cliente."""
    resumenes = {}
    for variable in variables:
        resumen = (
            df.groupby("Abandono_Cliente")[variable]
            .agg(["count", "mean", "median", "std"])
            .round(2)
        )
        print(f"\nResumen descriptivo de {variable}")
        print(resumen)
        resumenes[variable] = resumen
    return resumenes


def graficos_numericas_por_abandono(df, variables=VARIABLES_NUMERICAS):
    """Boxplot y violin plot de cada variable numérica según Abandono_Cliente."""
    sns.set_theme(style="whitegrid")
    for variable in variables:
        _, ax = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(
            data=df,
            x="Abandono_Cliente",
            y=variable,
            hue="Abandono_Cliente",
            palette="pastel",
            legend=False,
            ax=ax[0],
        )
        ax[0].set_title(f"Boxplot de {variable} por Abandono")

        sns.violinplot(
            data=df,
            x="Abandono_Cliente",
            y=variable,
            hue="Abandono_Cliente",
            palette="pastel",
            legend=False,
            ax=ax[1],
        )
        ax[1].set_title(f"Violin Plot de {variable} por Abandono")

        for eje in ax:
            eje.set_xlabel("Abandono (0 = No, 1 = Sí)")
            eje.set_ylabel(variable)

        plt.tight_layout()
        plt.show()


def curtosis(df, variables=VARIABLES_NUMERICAS):
    """Curtosis de las variables numéricas (ignora valores faltantes)."""
    return pd.DataFrame(
        {
            "Variable": variables,
            "Curtosis": [df[v].dropna().kurt() for v in variables],
        }
    )


def matriz_covarianza(df, variables=VARIABLES_NUMERICAS):
    """Calcula y grafica la matriz de covarianza de las variables numéricas."""
    matriz_cov = df[variables].cov()

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_cov, annot=True, fmt=".2f", cmap="Pastel1", linewidths=0.5)
    plt.title("Matriz de Covarianza")
    plt.tight_layout()
    plt.show()

    return matriz_cov


def tablas_frecuencia(df, variables=VARIABLES_CATEGORICAS):
    """Tablas de frecuencia absoluta, relativa y acumulada por variable."""
    tablas = {}
    for variable in variables:
        tabla = df[variable].value_counts().to_frame("Frecuencia absoluta")
        tabla.index = _etiquetas(df, variable, tabla.index)
        tabla["Frecuencia relativa (%)"] = (
            tabla["Frecuencia absoluta"] / tabla["Frecuencia absoluta"].sum() * 100
        ).round(2)
        tabla["Frecuencia acumulada"] = tabla["Frecuencia absoluta"].cumsum()
        tabla["Frecuencia relativa acumulada (%)"] = (
            tabla["Frecuencia relativa (%)"].cumsum().round(2)
        )

        print(f"TABLA DE FRECUENCIAS: {variable}")
        display(tabla)
        tablas[variable] = tabla
    return tablas


def matriz_correlacion_categoricas(df, variables=VARIABLES_CATEGORICAS):
    """Matriz de correlación de las variables categóricas codificadas."""
    df_corr = df[variables].copy()
    for col in df_corr.columns:
        df_corr[col] = df_corr[col].astype("category").cat.codes
    matriz_corr = df_corr.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="Pastel1", linewidths=0.5)
    plt.title("Matriz de Correlación de Variables Categóricas")
    plt.tight_layout()
    plt.show()

    return matriz_corr


def graficos_bivariados_abandono(df, variables=VARIABLES_CATEGORICAS[:-1]):
    """Barras de cada variable categórica cruzada con Abandono_Cliente."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    axes = axes.flatten()

    for i, variable in enumerate(variables):
        tabla = pd.crosstab(df[variable], df["Abandono_Cliente"])
        tabla.index = _etiquetas(df, variable, tabla.index)

        ax = tabla.plot(
            kind="bar",
            ax=axes[i],
            color=["#A8DADC", "#F4A6A6"],
            width=0.8,
        )
        ax.set_title(f"{variable} vs Abandono_Cliente", fontsize=12, pad=10)
        ax.set_xlabel(variable)
        ax.set_ylabel("Frecuencia")

        totales_categoria = tabla.sum(axis=1)
        for container in ax.containers:
            for j, barra in enumerate(container):
                altura = barra.get_height()
                if altura > 0:
                    porcentaje = altura / totales_categoria.iloc[j] * 100
                    ax.text(
                        barra.get_x() + barra.get_width() / 2,
                        altura,
                        f"{int(altura)}\n({porcentaje:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.legend(title="Abandono Cliente", labels=["No", "Sí"])
        ax.tick_params(axis="x", labelrotation=30)

    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def Analisis_Estadistico(df):
    """Ejecuta todo el análisis estadístico sobre el dataset ya limpio."""
    grafico_distribucion_categoricas(df)
    boxplots_numericas(df)
    resumenes = resumen_por_abandono(df)
    graficos_numericas_por_abandono(df)
    tabla_curtosis = curtosis(df)
    print(tabla_curtosis)
    matriz_cov = matriz_covarianza(df)
    print("Matriz de Covarianza")
    print(matriz_cov)
    tablas = tablas_frecuencia(df)
    matriz_corr = matriz_correlacion_categoricas(df)
    graficos_bivariados_abandono(df)

    return {
        "resumen_por_abandono": resumenes,
        "curtosis": tabla_curtosis,
        "matriz_covarianza": matriz_cov,
        "tablas_frecuencia": tablas,
        "matriz_correlacion": matriz_corr,
    }
