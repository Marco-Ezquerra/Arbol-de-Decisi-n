# arbol.py
import numpy as np
import pandas as pd

class Nodo:
    def __init__(self, datos: pd.DataFrame, nombre: str = "raiz", profundidad: int = 0, classes_=None, target_col: str = "deporte",    
        feature_cols: list | None = None,max_thresholds_per_feature: int | None = None, min_samples_leaf: int = 1  ):

        # Atributos de datos
        self.datos = datos
        self.nombre = nombre
        self.profundidad = profundidad
        self.n_muestras = len(datos)

        self.target_col = target_col          # CAMBIO
        if feature_cols is None:              # NUEVO (inferencia por defecto)
            self.feature_cols = [c for c in datos.columns if c != target_col]
        else:
            self.feature_cols = list(feature_cols)
        
        self.max_thresholds_per_feature = max_thresholds_per_feature   #numero maximo de cortes
        self.min_samples_leaf = int(min_samples_leaf)
        # Atributos de estado
        self.gini = None
        self.es_hoja = False
        self.prediccion = None

        # Atributos de división
        self.feature = None
        self.threshold = None
        self.izquierda = None
        self.derecha = None

        if classes_ is None:
            # en la raíz las inferimos; en hijos se heredarán                 #CLASE PADRE
            self.classes_ = np.array(sorted(self.datos[self.target_col].unique()))  #nos da la cantidad de posibles respuestas ordenadas de la variable objetivo Y 
        else:                                                                    #El objetivo de esto es que TODAS LAS HOJAS devuelvan las probabilidades en el mismo 
                                                                                 #orden de clases
            
            
            self.classes_ = np.array(classes_)  ##---------->Para los HIJOS simlemente le devolvemos los atributos del padre
       
        #CONTENDEDORES. Cuando el Nodo se convierta en HOja se activaran estos contadores

        self.class_counts_ = None   # número de muestras de entrenamiento de cada clase que cayeron en esa hoja.
        self.class_proba_  = None   # esas cuentas nromalizadas para dar el VECTOR DE PROBABILIDADES

        # Calcular gini inicial
        self.calcular_gini()

    def calcular_gini(self):
        y = self.datos[self.target_col]
        if len(y) == 0:
            self.gini = 0.0
            return
        proporciones = np.unique(y, return_counts=True)[1] / len(y)
        self.gini = 1.0 - float(np.sum(proporciones ** 2))
    
    def _marcar_hoja(self):
        self.es_hoja = True
        y = self.datos[self.target_col]

        if len(y) == 0:
            self.prediccion = None
            K = len(self.classes_)
            self.class_counts_ = np.zeros(K, dtype=int)
            self.class_proba_  = np.zeros(K, dtype=float)
            return

        # clase mayoritaria
        self.prediccion = y.mode()[0]

        # conteos alineados con self.classes_
        vals, cnts = np.unique(y, return_counts=True)  # vals: valores unicos ordenados de y   cnts: return_counts=True devuelve la frecuencia de cada valor
        
        mapa = {}
        for v, c in zip(vals, cnts):     #v --> val    # c----> conts   #junta valores y frecuencias (valor, frecuencia) y lo pasamos a un diccionario
            mapa[v] = c
         
        
        counts = np.array([mapa.get(c, 0) for c in self.classes_], dtype=int) #para cada clase c del conjuto de clases esperadas si c existe en el diccionario devuelve
        #la frecuencia de la clase esperada
        # importante que devuelva el 0 si c no pertenece al diccionario para representar a todas las clases

        self.class_counts_ = counts
        self.class_proba_  = counts / counts.sum()

        


    def encontrar_mejor_split(self):
        """Encuentra la mejor división; si no hay válida, marca hoja."""
        mejor_gini = float('inf')
        mejor_feature = None
        mejor_threshold = None

        # Si hay muy pocas muestras, convertir en hoja
        if len(self.datos) < 3:
            self._marcar_hoja()

            return None

        # Probar cada característica (sin midpoints, como acordamos)
        for feature in self.feature_cols:
            X = self.datos[feature]
            y = self.datos[self.target_col]
            valores_unicos = np.unique(X.to_numpy())

            if self.max_thresholds_per_feature is not None and len(valores_unicos) > self.max_thresholds_per_feature:
            # cuantiles uniformes en (0,1); evitamos extremos 0 y 1
                q = np.linspace(0, 1, num=self.max_thresholds_per_feature + 2)[1:-1]
            # usamos cuantiles sobre X (no sobre valores_unicos) para representar la distribución real
                x_arr = X.to_numpy()
                try:
                    candidates = np.unique(np.quantile(X.to_numpy(), q, method="linear"))
                except TypeError:
                    candidates = np.unique(np.quantile(x_arr, q, interpolation="linear"))
            # por seguridad, nos quedamos dentro del rango observado
                candidate_thresholds = candidates
            else:
                candidate_thresholds = valores_unicos
            
            if len(candidate_thresholds) < 2:
                continue

            for threshold in candidate_thresholds:
                mask_left  = (X <= threshold)
                mask_right = ~mask_left

    # CAMBIO mínimo → exigir mínimo en cada lado
                if mask_left.sum() < self.min_samples_leaf or mask_right.sum() < self.min_samples_leaf:
                    continue

                izquierda = y[mask_left]
                derecha   = y[mask_right]
                n = len(y)
                gini_izq = 1 - np.sum((np.unique(izquierda, return_counts=True)[1] / len(izquierda)) ** 2)
                gini_der = 1 - np.sum((np.unique(derecha,   return_counts=True)[1] / len(derecha))   ** 2)
                gini_ponderado = (len(izquierda)/n) * gini_izq + (len(derecha)/n) * gini_der

                if gini_ponderado < mejor_gini:
                    mejor_gini = gini_ponderado
                    mejor_feature = feature
                    mejor_threshold = threshold

        self.feature = mejor_feature
        self.threshold = mejor_threshold
        return mejor_gini if self.feature is not None else None

    def dividir_nodo(self):
        """Divide datos según la mejor división encontrada."""
        return (
            self.datos[self.datos[self.feature] <= self.threshold],
            self.datos[self.datos[self.feature] >  self.threshold],
        )

    def construir_subarbol(self, profundidad_max: int = 6):
        """Construye el subárbol recursivamente desde este nodo."""
        # Criterios de parada
        if (self.profundidad == profundidad_max or
            self.gini == 0.0 or
            len(self.datos) < max(2, 2 * self.min_samples_leaf)):
            
            self._marcar_hoja()


            return    # MUY IMPORTANTE ----------------------------> vuelve al punto  EXACTO desde el cual fue llamada esa función, para no perder el camino del arbol
        # Buscar split
        resultado_split = self.encontrar_mejor_split()

        # Si no hay split válido, hoja
        if resultado_split is None or self.feature is None:
            self._marcar_hoja()

            return

        # Dividir y crear hijos
        datos_izq, datos_der = self.dividir_nodo()
        self.izquierda = Nodo(datos_izq, f"{self.nombre}_izq", self.profundidad + 1,classes_=self.classes_,target_col=self.target_col,feature_cols=self.feature_cols,
        max_thresholds_per_feature=self.max_thresholds_per_feature,min_samples_leaf=self.min_samples_leaf)  #hereda classes_, columnas target y features y el maximo de divisiones, y minimo de samples para crear nodo terminal
        self.derecha   = Nodo(datos_der, f"{self.nombre}_der", self.profundidad + 1,classes_=self.classes_,target_col=self.target_col,feature_cols=self.feature_cols,
        max_thresholds_per_feature=self.max_thresholds_per_feature,min_samples_leaf=self.min_samples_leaf)  #hereda classes_ que corresponde a el atributo del nodo raiz
        
        #si no ponemos las columnas de features e Y objetivo hay que ponerlas como paremtros de entrada a ala clase nodo

    
        # Recursión
        self.izquierda.construir_subarbol(profundidad_max)
        self.derecha.construir_subarbol(profundidad_max)

    # --------- Predicción ----------
    #sino que usa la topología y parámetros del árbol que quedaron fijados tras e
    # l entrenamiento. Cada predicción es determinista e independiente.
    #EN EL ENTRENAMIENTO SE FIJAN LOS CAMINOS A SEGUIR, LOS DATOS A PREDECIR SEGUN EL CAMINO
    def predecir_fila(self, fila: pd.Series):
        """Predice la clase para una fila."""        #hacemos algo parecido en la subrutina get_hoja
        if self.es_hoja or self.feature is None:
            return self.prediccion 
        if fila[self.feature] <= self.threshold:
            return self.izquierda.predecir_fila(fila) if self.izquierda is not None else self.prediccion
        else:
            return self.derecha.predecir_fila(fila)   if self.derecha   is not None else self.prediccion

    def predecir(self, df_X: pd.DataFrame) -> pd.Series:
    # """
    # Predice para un DataFrame de características.

    # Parámetros
    # ----------
    # df_X : pd.DataFrame
    #     Conjunto de entrada con las columnas que usa el árbol para dividir
    #     (en este ejemplo, 'edad' y 'horas_estudio').

    # Devuelve
    # --------
    # pd.Series
    #     Serie de longitud len(df_X) con la clase predicha para cada fila,
    #     manteniendo el mismo índice que df_X.
    # """
    # apply(func, axis=1) aplica la función a CADA FILA del DataFrame (axis=1 = por filas).
    # self.predecir_fila recibe una fila (pd.Series) y recorre el árbol hasta una hoja,
    # devolviendo la etiqueta predicha ("sí"/"no") para esa fila.
    # El resultado de apply es un pd.Series con una predicción por fila.
        return df_X.apply(self.predecir_fila, axis=1)

    def get_hoja(self, fila: pd.Series): #---->Le metemso una fila del dataset y esta recorre el arbol hasta encontrar un Nodo hoja
        
        nodo = self                     #------------> Empezamos por el nodo raiz     

        while not (nodo.es_hoja or nodo.feature is None):   #-----> No paramos hasta encontrar un NODO TERMINAL/HOJA
            if fila[nodo.feature] <= nodo.threshold:
                if nodo.izquierda is None: break   #si no exite hijo paramos, estamos en un NODO HOJA!!
                nodo = nodo.izquierda
            else:
                if nodo.derecha is None: break
                nodo = nodo.derecha
        return nodo
    
    def proba_fila(self, fila:pd.Series)->np.ndarray:

        hoja= self.get_hoja(fila)

        #-------------------------------Si algun nodo no tiene las stats-------------------------------
        if hoja.class_counts_ is None or hoja.class_proba_ is None:
            proba = np.zeros(len(self.classes_), dtype=float)
            if hoja.prediccion is not None:
                i = np.where(self.classes_ == hoja.prediccion)[0]
                if len(i): proba[i[0]] = 1.0
            return proba
        #----------------------------------------------------------------------------------------------------
        
        
        # Nota:
    # El parámetro alpha controla el "suavizado de Laplace".
    # - Con alpha=0 (por defecto) las probabilidades se calculan como frecuencias puras,
    #   es decir P(k) = n_k / n. Esto replica sklearn y funciona bien en general.
    # - El inconveniente es que en hojas con muy pocas muestras pueden salir 0.0 o 1.0 exactos,
    #   lo que da una confianza excesiva y puede ser problemático si luego se usan log-probabilidades.
    # - Con alpha>0 añadimos pseudocuentas: P(k) = (n_k + alpha) / (n + K*alpha).
    #   Esto evita ceros/unos exactos y mejora la calibración en hojas pequeñas.
    # - No usar alpha>0 no da errores numéricos, simplemente las probabilidades pueden
    #   estar "sobreajustadas" cuando las hojas son muy pequeñas.

        
        
        return hoja.class_proba_.astype(float) #devuelve el vector de probabilidades


    def predict_proba(self, df_X: pd.DataFrame)-> np.ndarray:   #apply va  devolver una seri de arrays
        probas = df_X.apply(self.proba_fila, axis=1)
        return np.vstack(probas.values)  #me lo pasa a matris (n_muestras, n_clases)