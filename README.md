# Comparativa entre t-SNE y MDS en Reducción de Dimensionalidad de Datos

Repositorio del Trabajo de Fin de Máster (TFM) que presenta una investigación comparativa y sistemática de técnicas de reducción de dimensionalidad, con un enfoque principal en la preservación de la estructura local de los datos.

## 📌 Métodos Analizados

Este estudio implementa y evalúa cuatro métodos fundamentales de reducción de dimensionalidad para proyectar datos a un espacio 2D:

1.  **t-SNE**: La implementación optimizada de Barnes-Hut (`Rtsne`), que destaca por su capacidad para visualizar agrupaciones locales.
2.  **MDS Local**: Una variante de `smacof` que prioriza la estructura local mediante una función de *stress* ponderada por vecindad (`k`-vecinos).
3.  **Sammon Mapping**: Un caso particular de MDS con un esquema de pesos ($w_{ij} = 1/\delta_{ij}$) que enfatiza la preservación de distancias cortas.
4.  **MDS Clásico**: También conocido como Principal Coordinates Analysis (PCoA), sirve como método de referencia para la preservación de la estructura métrica global.

## 📊 Datasets

Los experimentos se realizan sobre un conjunto diverso de **8 datasets** para evaluar el rendimiento de los algoritmos en diferentes escenarios de complejidad, tamaño y dimensionalidad:

#### Datasets Públicos

* **Iris**: Dataset tabular de referencia con estructura simple.
* **Wine**: Datos tabulares de análisis químico.
* **Olivetti Faces**: Imágenes de rostros en escala de grises.
* **Mushroom**: Dataset categórico con atributos de hongos.
* **Adult Census Income**: Datos de censo con variables mixtas.
* **MNIST**: Imágenes de dígitos manuscritos (20,000 muestras).
* **Fashion-MNIST**: Imágenes de artículos de moda (20,000 muestras).

#### Dataset Sintético

* **Datos Sintéticos**: Un conjunto de datos generado con clústeres gaussianos para evaluar el rendimiento en un entorno controlado y analizar el impacto de los hiperparámetros.

## 🛠️ Metodología

1.  **Preprocesado**: Se aplica un pipeline de preprocesado común (`preprocess.R`) a todos los datasets, que incluye normalización Z-score y eliminación de outliers.
2.  **Reducción de Dimensionalidad**: Cada uno de los 4 métodos se aplica a los datos preprocesados. Para los datos sintéticos, se exploran múltiples configuraciones de hiperparámetros.
3.  **Evaluación Cuantitativa**: Se utilizan las siguientes métricas (`metrics.R`) para una evaluación objetiva:
    * **Preservación Local**: `Trustworthiness` y `Continuity`.
    * **Preservación de Ranking Global**: Correlación de `Spearman`.
    * **Similitud Geométrica**: `Análisis de Procrustes` ($m^2$) para medir la congruencia en la forma de los mapas 2D producidos por cada par de métodos.

## 📁 Estructura del Repositorio

```
.
├── data/                 # Directorio para almacenar los datasets (no incluido en el repo)
├── proclustes/           # Scripts para el análisis de Procrustes por dataset
│   ├── Adult.R
│   ├── iris.R
│   └── ...               
├── results/              # Directorio (creado por los scripts) para guardar las métricas y resultados
├── src_datasets/         # Scripts principales, uno por cada dataset
│   ├── Iris/
│   ├── MNIST/
│   └── ...
├── metrics.R             # Implementación de las métricas de evaluación
├── plot_theme.R          # Tema personalizado para los gráficos ggplot2
├── preprocess.R          # Funciones para el preprocesado de datos
├── README.md             # Este archivo
```

## 🚀 Cómo Replicar los Experimentos

#### 1. Prerrequisitos
* Tener instalado R (versión 4.0 o superior).


#### 2. Ejecución

Puedes ejecutar los análisis para cada dataset de forma individual con Rscript fichero.R

