# 🌳 Árbol de Decisión “from scratch” en Python (clase `Nodo`)

Este proyecto implementa **desde cero** un **árbol de decisión** en Python usando únicamente `numpy` y `pandas`.  
El objetivo no es competir con librerías como `scikit-learn`, sino **aprender a fondo cómo se construyen y funcionan los árboles de decisión**:  
- Cómo se mide la impureza (Gini).  
- Cómo se eligen los thresholds.  
- Cómo se crean las hojas y se calculan las probabilidades.  

Gracias a esta base, se entiende mucho mejor el funcionamiento de algoritmos como **Random Forest** o **Gradient Boosting**.  

---

## ✨ Características principales

✔️ Criterio **Gini** como medida de impureza  
✔️ Control de complejidad con:  
- `profundidad_max` (profundidad máxima)  
- `min_samples_leaf` (mínimo de muestras en cada hoja)  
- `max_thresholds_per_feature` (nº máximo de cortes por feature, usando cuantiles)  
✔️ Métodos de predicción:  
- `predecir` → clase mayoritaria  
- `predict_proba` → probabilidades por clase  
✔️ Soporta datasets binarios y multiclase  
✔️ Código 100% Python, claro y educativo  

---



## 🧱 Requisitos

- Python 3.9+
- Librerías:  
  ```bash
  pip install numpy pandas scikit-learn


## 📂 Archivos incluidos

📌 **arbol_modulo.py**
Contiene la implementación pura de la clase Nodo.
Aquí está toda la lógica de construcción del árbol:

Cálculo de impureza (Gini).

Selección del mejor split.

Creación de hojas y almacenamiento de probabilidades.

Métodos predecir y predict_proba.

Este archivo es la base principal del proyecto, desarrollado manualmente para entender cómo funciona un árbol de decisión desde dentro.

📌 **comparacion_manual_vs_libreria.py**
Script generado a partir de un promt **ChatGPT**.

Su función es comparar el rendimiento del árbol manual (Nodo) con el de DecisionTreeClassifier de scikit-learn.

Incluye:

Entrenamiento y evaluación de ambos modelos en datasets de ejemplo (breast_cancer, iris).

Métricas (Accuracy, AUC, LogLoss, Brier score).

Comparaciones en múltiples seeds (para analizar varianza).

Este archivo no forma parte del desarrollo central del árbol, sino que sirve como validador externo para comprobar que los resultados del Nodo son consistentes con los de scikit-learn.
