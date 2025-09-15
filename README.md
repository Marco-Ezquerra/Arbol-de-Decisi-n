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
