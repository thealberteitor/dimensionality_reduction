# Comparativa entre t-SNE y MDS en Reducción de Dimensionalidad

Este repositorio contiene el código, datos y resultados del Trabajo de Fin de Máster (TFM) **Comparativa entre t‑SNE y MDS en la Reducción de Dimensionalidad de Datos**. Se implementan y comparan cuatro métodos de reducción de dimensionalidad:

- **t‑SNE** (Barnes–Hut)  
- **MDS Clásico** (métrico, `cmdscale`)  
- **MDS No Métrico** (ordinal, `isoMDS`)  
- **Local‑MDS** (SMACOF con pesos locales)  

> **Nota:** Sammon Mapping se considera un caso particular de Local‑MDS (con pesos \(w_{ij}=1/\delta_{ij}\)).


## 📁 Estructura del repositorio
- Datasets Iris, Wine, Olivetti Faces y MNIST. 

