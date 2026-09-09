# Importar librerías necesarias para la limpieza del dataset
import pandas as pd 
import zipfile 

# ruta del archivo comprimido que contiene los datos
archivo = "archivo.zip"


# Abrir el archivo ZIP y cargar los CSVs en DataFrames
with zipfile.ZipFile(archivo) as z:
    # Información de competiciones
    eventos = pd.read_csv(z.open("Scraped_Event_Details.csv"))
    # Información de patinadores
    skaters = pd.read_csv(z.open("Scraped_Skater_Details.csv"))
    # Puntuaciones técnicas
    tecnicos = pd.read_csv(z.open("Scraped_Technical_Scores.csv"))
    # Puntuaciones de componentes artísticos
    componentes = pd.read_csv(z.open("Scraped_Component_Scores.csv"))

# Filtrar eventos de la ronda Qualifying
evento_qualifying = eventos[eventos["Round"] == "Qualifying"][
    ["Event_Index", "Region", "Group"]
]

# Combinar datos de patinadores con eventos de Qualifying
qualifying = skaters.merge(
    evento_qualifying,
    on="Event_Index",
    how="inner"
)

# Eliminar patinadores que se retiraron (WD = Withdrawn)
qualifying = qualifying[
    qualifying["Skater_Placement"] != "WD"
].copy()

# Convertir la posición del patinador a tipo numérico
qualifying["Skater_Placement"] = pd.to_numeric(
    qualifying["Skater_Placement"]
)

# Convertir el orden de patinaje a tipo numérico
qualifying["Skate_Order"] = pd.to_numeric(
    qualifying["Skate_Order"]
)

# Filtrar eventos de Final con programa corto (Short Program) para identificar finalistas
evento_final = eventos[
    (eventos["Round"] == "Final") &
    (eventos["Program_Type"] == "Short Program")
][["Event_Index", "Region"]]

# Combinar datos de patinadores con eventos finales
finalistas = skaters.merge(
    evento_final,
    on="Event_Index",
    how="inner"
)

# Crear diccionario con nombre de patinadores finalistas por región
finalistas_por_region = {
    region: set(grupo["Skater_Name"])
    for region, grupo in finalistas.groupby("Region")
}

# Crear columna target: 1 si el patinador calificó a Final, 0 si no
qualifying["Qualified"] = qualifying.apply(
    lambda fila: 1
    if fila["Skater_Name"] in finalistas_por_region.get(
        fila["Region"], set()
    )
    else 0,
    axis=1
)

# Resumir estadísticas técnicas por evento y posición del patinador
tecnicos_resumen = tecnicos.groupby(
    ["Event_Index", "Skater_Placement"]
).agg(

    # Cantidad total de elementos técnicos
    Element_Count=(
        "Element_Number",
        "count"
    ),

    # Valor base total de todos los elementos
    Base_Value_Total=(
        "Base_Value",
        "sum"
    ),

    # Valor base promedio de los elementos
    Base_Value_Avg=(
        "Base_Value",
        "mean"
    ),

    # GOE (Grado de Ejecución) promedio
    GOE_Avg=(
        "GOE",
        "mean"
    ),

    # GOE mínimo (peor ejecución)
    GOE_Min=(
        "GOE",
        "min"
    ),

    # GOE máximo (mejor ejecución)
    GOE_Max=(
        "GOE",
        "max"
    ),

    # Cantidad de elementos con GOE negativo
    Negative_GOE_Count=(
        "GOE",
        lambda x: int((x < 0).sum())
    ),

    # Cantidad de saltos en la segunda mitad del programa
    Second_Half_Count=(
        "FreeSkate2ndHalfJump_Flag",
        lambda x: int((x == "x").sum())
    )

).reset_index()

# Reorganizar los componentes artísticos como columnas individuales
componentes_resumen = componentes.pivot_table(
    index=["Event_Index", "Skater_Placement"],
    columns="Program_Components",
    values="PanelScores_PC",
    aggfunc="first"
).reset_index()

componentes_resumen.columns.name = None

# Renombrar columnas 
componentes_resumen = componentes_resumen.rename(
    columns={
        "Skating Skills": "Skating_Skills",        # Habilidades de patinaje
        "Performance": "Performance",              # Desempeño general
        "Composition": "Composition",              # Composición
        "Interpretation of the Music":
            "Interpretation_Music"                  # Interpretación de la música
    }
)

# Combinar datos de Qualifying con resumen técnico
dataset = qualifying.merge(
    tecnicos_resumen,
    on=["Event_Index", "Skater_Placement"],
    how="left"
)

# Combinar con resumen de componentes artísticos
dataset = dataset.merge(
    componentes_resumen,
    on=["Event_Index", "Skater_Placement"],
    how="left"
)

# Seleccionar columnas relevantes para el modelo de predicción
columnas_modelo = [
    # Características técnicas (cualidad de los elementos)
    "Element_Count",           # Cantidad de elementos
    "Base_Value_Total",        # Valor base total
    "Base_Value_Avg",          # Valor base promedio
    "GOE_Avg",                 # GOE promedio
    "GOE_Min",                 # GOE mínimo
    "GOE_Max",                 # GOE máximo
    "Negative_GOE_Count",      # Cantidad de GOE negativos
    "Second_Half_Count",       # Saltos en segunda mitad

    # Características de componentes artísticos
    "Skating_Skills",          # Habilidades de patinaje
    "Performance",             # Desempeño
    "Composition",             # Composición
    "Interpretation_Music",    # Interpretación

    # Deducciones por penalizaciones
    "Deductions_Falls",        # Caídas
    "Deductions_CostumeOrPropFailure",  # Fallos de vestuario/prop
    "Deductions_TimeViolation",         # Violación de tiempo

    # Variable objetivo (target)
    "Qualified"                # 1 si calificó a la fiinal, 0 si no
]

# Crear dataset final con solo las columnas seleccionadas
dataset_final = dataset[columnas_modelo].copy()

# Información del dataset final
print(f"Forma del dataset: {dataset_final.shape}")  # Número de filas y columnas
print(f"\nValores faltantes:\n{dataset_final.isnull().sum()}")  # Contar valores nulos
print(f"\nDistribución de la variable target:\n{dataset_final['Qualified'].value_counts()}")  # Contar calificados vs no calificados
print(f"\nPrimeras filas del dataset:\n{dataset_final.head()}")

# Guardar dataset procesado en CSV
dataset_final.to_csv("figure_skating_dataset.csv", index=False)
print("\n✓ Dataset guardado en 'figure_skating_dataset.csv'")
