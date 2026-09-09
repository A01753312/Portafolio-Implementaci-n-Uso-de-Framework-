# 1. Importar librerías necesarias y datos
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report)
# 2. Cargar el dataset
dataset = pd.read_csv("figure_skating_dataset.csv")

print("Dataset cargado exitosamente. Primeras filas:")
print("Forma del dataset:", dataset.shape)

# separar características y variable objetivo
X = dataset.drop(columns=["Qualified"])
Y = dataset["Qualified"]

#3. Dividir el dataset en conjunto de entrenamiento y prueba
# 80% entrenamiento, 20% prueba

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("\n DIVISIÓN DEL DATASET: ")

print("Datos de entrenamiento:", len(Xtrain))
print("Datos de prueba:", len(Xtest))

print("\nDistribución Train:")
print(Ytrain.value_counts().sort_index())

print("\nDistribución Test:")
print(Ytest.value_counts().sort_index())

#4. Crear y entrenar el modelo Random Forest

modelo = RandomForestClassifier(
    n_estimators=15,
    max_depth=5,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42
)

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(Xtrain, Ytrain)

print("\nRandom Forest entrenado correctamente.")
print("Número de árboles generados:", len(modelo.estimators_))

# 5. Realizar predicciones 
predicciones = modelo.predict(Xtest)
print("\nPredicciones realizadas correctamente")

# 5.1 Evaluar el modelo 


matriz_confusion = confusion_matrix(
    Ytest,
    predicciones
)

accuracy = accuracy_score(
    Ytest,
    predicciones
)

precision = precision_score(
    Ytest,
    predicciones
)

recall = recall_score(
    Ytest,
    predicciones
)

f1 = f1_score(
    Ytest,
    predicciones
)


print("\n--- Resultados Random Forest con Framework ---")

print("\nMatriz de confusión:")
print(matriz_confusion)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nInforme de clasificación:")

print(
    classification_report(
        Ytest,
        predicciones,
        target_names=[
            "No calificó",
            "Calificó"
        ],
        digits=4
    )
)

# 6. Optimización de hiperparámetros con GridSearchCV

# Configuraciones que se desean evaluar
param_grid = {

    "n_estimators": [
        15,
        50,
        100
    ],

    "max_depth": [
        4,
        5,
        6
    ],

    "min_samples_split": [
        2,
        4
    ],

    "max_features": [
        2,
        3,
        4
    ]
}


# Crear un Random Forest base para la búsqueda
modelo_busqueda = RandomForestClassifier(
    random_state=42
)


# GridSearchCV prueba diferentes combinaciones
# usando únicamente el conjunto de entrenamiento
grid_search = GridSearchCV(
    estimator=modelo_busqueda,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)


print("\nBuscando mejores hiperparámetros...")


# Entrenamiento de las diferentes configuraciones
grid_search.fit(
    Xtrain,
    Ytrain
)


print("\n--- Mejor configuración encontrada ---")

print(
    "Mejores parámetros:",
    grid_search.best_params_
)

print(
    "Mejor F1 promedio en validación:",
    round(grid_search.best_score_, 4)
)

#6.1 Evaluar el modelo optimizado con los mejores hiperparámetros

mejor_modelo = grid_search.best_estimator_
# Realizar predicciones sobre el conjunto de prueba
predicciones_mejor_modelo = mejor_modelo.predict(
    Xtest
)


# Calcular métricas
matriz_mejor = confusion_matrix(
    Ytest,
    predicciones_mejor_modelo
)

accuracy_mejor = accuracy_score(
    Ytest,
    predicciones_mejor_modelo
)
precision_mejor = precision_score(
    Ytest,
    predicciones_mejor_modelo
)

recall_mejor = recall_score(
    Ytest,
    predicciones_mejor_modelo
)

f1_mejor = f1_score(
    Ytest,
    predicciones_mejor_modelo
)


print("\n--- Resultados del mejor modelo ---")

print("\nMatriz de confusión:")
print(matriz_mejor)

print(
    "\nAccuracy:",
    round(accuracy_mejor, 4)
)
print(
    "Precision:",
    round(precision_mejor, 4)
)

print(
    "Recall:",
    round(recall_mejor, 4)
)

print(
    "F1 Score:",
    round(f1_mejor, 4)
)

# 7 Gráficas de resultados 
#7.1 Matriz de confusión del mejor modelo
plt.figure(figsize=(6, 5))

plt.imshow(matriz_mejor)

plt.title("Matriz de Confusión - Random Forest Optimizado")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")

plt.xticks(
    [0, 1],
    ["No calificó (0)", "Calificó (1)"]
)

plt.yticks(
    [0, 1],
    ["No calificó (0)", "Calificó (1)"]
)
# Escribir los valores dentro de la matriz
for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            matriz_mejor[i][j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.colorbar()
plt.tight_layout()

plt.savefig(
    "matriz_confusion_framework.png",
    dpi=300
)

plt.close()

# 7.3 Matriz de confusión del modelo base

plt.figure(figsize=(6, 5))

plt.imshow(matriz_confusion)

plt.title("Matriz de Confusión - Random Forest Base")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")

plt.xticks(
    [0, 1],
    ["No calificó (0)", "Calificó (1)"]
)

plt.yticks(
    [0, 1],
    ["No calificó (0)", "Calificó (1)"]
)
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            matriz_confusion[i][j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "matriz_confusion_base.png",
    dpi=300
)

plt.close()

#7.2 Comparación de métricas entre el modelo base y el optimizado


metricas = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

resultados_base = [
    accuracy,
    precision,
    recall,
    f1
]

resultados_optimizado = [
    accuracy_mejor,
    precision_mejor,
    recall_mejor,
    f1_mejor
]
# Posiciones de las barras
x = list(range(len(metricas)))

ancho = 0.35

x_base = [
    posicion - ancho / 2
    for posicion in x
]

x_optimizado = [
    posicion + ancho / 2
    for posicion in x
]


plt.figure(figsize=(8, 5))

barras_base = plt.bar(
    x_base,
    resultados_base,
    width=ancho,
    label="Modelo base"
)
barras_optimizado = plt.bar(
    x_optimizado,
    resultados_optimizado,
    width=ancho,
    label="Modelo optimizado"
)

plt.xticks(
    x,
    metricas
)

plt.ylabel("Valor")
plt.title("Comparación de métricas - Random Forest")
plt.ylim(0, 1)

plt.legend()


# Mostrar valores sobre las barras del modelo base
for barra, valor in zip(
    barras_base,
    resultados_base
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.01,
        f"{valor:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )
# Mostrar valores sobre las barras del modelo optimizado
for barra, valor in zip(
    barras_optimizado,
    resultados_optimizado
):

    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.01,
        f"{valor:.4f}",
        ha="center",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    "comparacion_modelos_framework.png",
    dpi=300
)

plt.close()

print("\nGráficas generadas correctamente:")
print("- matriz_confusion_framework.png")
print("- matriz_confusion_base.png")
print("- comparacion_modelos_framework.png")

# 8 Análisis de overfitting 
# Accuracy de entrenamiento
predicciones_train = mejor_modelo.predict(Xtrain)

accuracy_train = accuracy_score(
    Ytrain,
    predicciones_train
)

# Accuracy de prueba
accuracy_test = accuracy_mejor

# Diferencia entre ambos
diferencia = accuracy_train - accuracy_test


print("\n--- Análisis de Overfitting ---")

print(
    "Accuracy entrenamiento:",
    round(accuracy_train, 4)
)

print(
    "Accuracy prueba:",
    round(accuracy_test, 4)
)
print(
    "Diferencia:",
    round(diferencia, 4)
)
# 8.1 gráfica train vs test 

conjuntos = [
    "Entrenamiento",
    "Prueba"
]

valores_accuracy = [
    accuracy_train,
    accuracy_test
]

plt.figure(figsize=(6, 5))

barras = plt.bar(
    conjuntos,
    valores_accuracy
)

plt.title(
    "Accuracy de Entrenamiento vs Prueba"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)


# Mostrar valores sobre las barras
for barra, valor in zip(
    barras,
    valores_accuracy
):

    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.02,
        f"{valor:.4f}",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    "train_vs_test_framework.png",
    dpi=300
)

plt.close()


print(
    "- train_vs_test_framework.png"
)
