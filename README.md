# Implementación de Random Forest con Framework

## Descripción

Este proyecto corresponde a la implementación de un algoritmo de Machine Learning utilizando un framework. Se utilizó el algoritmo **Random Forest** mediante la biblioteca **Scikit-learn** para resolver un problema de clasificación relacionado con competencias de patinaje artístico.

El objetivo del modelo es predecir si una patinadora calificará o no a la siguiente etapa de la competencia a partir de características relacionadas con su desempeño técnico, artístico y deducciones.

La variable objetivo utilizada es:

- `0`: No calificó
- `1`: Calificó

---

## Dataset

Se utiliza el archivo:

`figure_skating_dataset.csv`

El dataset contiene:

- 398 observaciones
- 15 variables predictoras
- 1 variable objetivo (`Qualified`)

Las variables predictoras incluyen información sobre ejecución técnica, valores base, Grade of Execution (GOE), componentes del programa y deducciones.

### Distribución de clases

| Clase | Observaciones | Proporción |
|---|---:|---:|
| No calificó (0) | 258 | 64.82% |
| Calificó (1) | 140 | 35.18% |
| Total | 398 | 100% |

---

## División de los datos

Los datos se dividieron utilizando `train_test_split` de Scikit-learn.

Se utilizó:

- 80% para entrenamiento
- 20% para prueba
- `random_state=42`
- `stratify=Y`

La división obtenida fue:

| Conjunto | Total | Clase 0 | Clase 1 |
|---|---:|---:|---:|
| Entrenamiento | 318 | 206 | 112 |
| Prueba | 80 | 52 | 28 |

El uso de `stratify` permite mantener una distribución similar de las clases en ambos conjuntos.

---

## Framework utilizado

La implementación fue desarrollada utilizando **Scikit-learn**.

Las principales herramientas utilizadas fueron:

- `RandomForestClassifier`
- `train_test_split`
- `GridSearchCV`
- `confusion_matrix`
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `classification_report`

También se utilizaron:

- `pandas` para cargar y manipular el dataset.
- `matplotlib` para generar las gráficas de resultados.

---

## Modelo Random Forest base

Inicialmente se configuró un Random Forest con los siguientes hiperparámetros:

```python
RandomForestClassifier(
    n_estimators=15,
    max_depth=5,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42
)
