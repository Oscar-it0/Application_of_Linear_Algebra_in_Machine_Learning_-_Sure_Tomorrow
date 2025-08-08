#!/usr/bin/env python
# coding: utf-8

# # ¡Hola Oscar! 😊
# 
# Mi nombre es **Alejandro Castellanos** y hoy tengo el placer de ser el revisor de tu proyecto.
# 
# Voy a revisar todo tu código con detalle, buscando tanto los puntos fuertes como aquellos en los que podrías mejorar. Te dejaré comentarios a lo largo del notebook, destacando lo que has hecho bien y sugiriendo ajustes donde sea necesario. Si encuentro algún error, no te preocupes, te lo haré saber de forma clara y te daré información útil para que puedas corregirlo en la próxima iteración. Si en algún punto tienes comentarios, siéntete libre de dejarlos también.
# 
# 
# Encontrarás mis comentarios específicos dentro de cajas verdes, amarillas o rojas, es muy importante que no muevas, modifiques o borres mis comentarios, con el fin de tener un seguimiento adecuado de tu proceso:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>
# 
# A continuación te dejaré un comentario general con mi valoración del proyecto. **¡Mi objetivo es que sigas aprendiendo y mejorando con cada paso!**

# ----

# <div class="alert alert-block alert-success">
# <b>Comentario General del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Oscar, hiciste un trabajo muy completo y bien estructurado. Destacaste el impacto del escalado en kNN, aplicaste correctamente métricas como RMSE, R² y F1, y comparaste modelos de forma efectiva. Además, lograste una transformación algebraica sólida para evaluar datos ofuscados. ¡Excelente!
# 
# ¡Continúa por este camino y te deseo mucho éxito en tu próximo Sprint! 🚀
#  
# *Estado del Proyecto*: **Aprobado**
# 
# </div>

# ----

# Comentario inicial.
# 
# Por favor considerar que los comentarios por sección se encuentran al final de cada una de las
# secciones.

# # Descripción

# La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.
# - Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
# - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
# - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento u ofuscación de datos. Pero los datos deben protegerse de tal manera que no se vea afectada la calidad de los modelos de machine learning. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.
# 

# # Preprocesamiento y exploración de datos
# 
# ## Inicialización

# In[1]:


pip install scikit-learn --upgrade


# In[2]:


# Importación de librerías

import numpy as np
import pandas as pd

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import r2_score
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing

import math
from sklearn.model_selection import train_test_split

from IPython.display import display


# ---------------

# ## Carga de datos

# Carga los datos y haz una revisión básica para comprobar que no hay problemas obvios.

# In[3]:


# Importación de DF

df = pd.read_csv('/datasets/insurance_us.csv')


# Renombramos las columnas para que el código se vea más coherente con su estilo.

# In[4]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo has renombrado las columnas para que sean más consistentes y manejables, lo cual es una buena práctica para evitar errores en el manejo de datos.
# 
# 
# 
# </div>

# In[5]:


df.sample(10)


# In[6]:


df.info()


# In[7]:


# puede que queramos cambiar el tipo de edad (de float a int) aunque esto no es crucial

# escribe tu conversión aquí si lo deseas:

df['age'] = df['age'].astype('int64')


# In[8]:


# comprueba que la conversión se haya realizado con éxito

df.info()


# In[9]:


# ahora echa un vistazo a las estadísticas descriptivas de los datos.# ¿Se ve todo bien?


# In[10]:


# Revisión de filas duplicadas en dataframes
print("Filas duplicadas en df: ",df.index.duplicated().sum())


# In[11]:


# Forma de los dataframes
print("Filas y columnas en df: ",df.shape)


# In[12]:


# Verificar si hay valores NaN

hay_nan = df.isnull().any().any()
if hay_nan == True:
    print("El DataFrame contiene valores ausentes")
else:
    print("El DataFrame NO contiene valores ausentes")

# Verificar si hay valores inf
df_inf = np.isinf(df)
if df_inf.any().any():
    print("El DataFrame contiene valores infinitos")
else:
    print("El DataFrame NO contiene valores infinitos")


# In[13]:


# Descripción de datos

df.describe()


# In[14]:


sns.boxplot(data = df)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Muy buen trabajo usando funciones como `info`, `describe` y `sample`, esto te permite hacer una primera revisión de los datos, su estructura y contenido. 
#     
# Además hiciste una comprobación de datos faltantes, lo cual es clave para evitar errores o sesgos en el análisis de los dato
# 
# </div>

# -----------------------

# ## Análisis exploratorio de datos

# Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.

# In[15]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# De acuerdo, es un poco complicado detectar grupos obvios (clústeres) ya que es difícil combinar diversas variables simultáneamente (para analizar distribuciones multivariadas). Ahí es donde LA y ML pueden ser bastante útiles.

# Comentarios Preprocesamiento y exploración de datos.
# 
# * Importación de librerías
# * Importación de DF
#     * Renombramiento de columnas
#     * Conversión de tipo de datos
#     * Revisión de filas duplicadas
#     * Verificación de valores ausentes e infinitos
#     * Descripción de datos
#     * Distribución de datos: Se observa que la magnitud de la columna "income" puede afectar en el desempeño del modelo.

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Has creado correctamente el *pair-plot*. Es muy importante que este tipo de información la analises e interpretes para que no se queden como datos sin contexto. Por ejemplo, podemos ver que *age* tiene una distribución sesgada a la derecha, lo que indica que hay más personas jóvenes que mayores, mientras que *insurance_benefits* presenta una distribución concentrada en valores bajos (0 o 1 predominante).
# 
# </div>

# ------------------

# # Tarea 1. Clientes similares

# En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.
# 
# Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)
# 
# - Distancia entre vectores -> Distancia euclidiana
# - Distancia entre vectores -> Distancia Manhattan
# 
# Para resolver la tarea, podemos probar diferentes métricas de distancia.

# Escribe una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.
# 
# Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) o tu propia implementación.
# 
# Pruébalo para cuatro combinaciones de dos casos
# 
# - Escalado
#   - los datos no están escalados
#   - los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
# - Métricas de distancia
#   - Euclidiana
#   - Manhattan
# 
# Responde a estas preguntas:
# - ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?
# - ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

# In[16]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[17]:


def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar    :param n: número de objetos para los que se buscan los vecinos más cercanos    :param k: número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia    """

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# In[18]:


knn_test_e = get_knn(df, 2, 4, 'euclidean')
knn_test_e


# In[19]:


knn_test_m = get_knn(df, 2, 4, 'cityblock')
knn_test_m


# Escalar datos.

# In[20]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[21]:


df_scaled.sample(5)


# Ahora, vamos a obtener registros similares para uno determinado, para cada combinación

# In[22]:


knn_test_scld_e = get_knn(df_scaled, 2, 4, 'euclidean')
knn_test_scld_e


# In[23]:


knn_test_scld_m = get_knn(df_scaled, 2, 4, 'cityblock')
knn_test_scld_m


# Comentarios Tarea 1.
# 
# * Definición y aplicación de funciones:
#     * get_knn
#    
# * Escalamiento de datos con MaxAbsScaler

# Respuestas a las preguntas

# **¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?** 
# 
# * Las distancias obtenidas utilizando los datos originales para los 4 vecinos cambian segun los tipos de distancia "Euclidiana" y "Manhattan", sin embargo los vecinos son los mismos. Sin embargo al utilizar datos escalados los vecinos cambian, así también las distancias según los dos tipos evaluados.

# **¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?** 
# 
# * Son similares, sin embargo la distancia Euclidiana es por poco menor en los casos analizados.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Has hecho un buen trabajo identificando cómo el escalado de datos influye en los resultados del algoritmo kNN, especialmente al observar el cambio en los vecinos seleccionados al aplicar MaxAbsScaler. También es acertado cómo reconoces las diferencias sutiles entre las métricas de distancia evaluadas, destacando que, aunque los resultados son similares, la métrica Euclidiana tiende a producir distancias ligeramente menores.
# 
# </div>

# ------------------

# # Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

# En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

# Con el valor de `insurance_benefits` superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.
# 
# Instrucciones:
# 
# - Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) o tu propia implementación.
# - Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.
# 
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
# 
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$
# 
# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

# In[24]:


# сalcula el objetivo
df['insurance_benefits_received'] = df['insurance_benefits']>0
df['insurance_benefits_received'] = df['insurance_benefits_received'] * 1
df

# сalcula el objetivo scaled
df_scaled['insurance_benefits_received'] = df_scaled['insurance_benefits']>0
df_scaled['insurance_benefits_received'] = df_scaled['insurance_benefits_received'] * 1
df_scaled


# In[25]:


# comprueba el desequilibrio de clases con value_counts()

ibr = df['insurance_benefits_received'].value_counts()
ibr


# In[26]:


sns.barplot(df['insurance_benefits_received'], df['insurance_benefits_received'].value_counts(), y=df['insurance_benefits_received'])


# In[27]:


def get_knn_class(data, neighbors):
    features = data.drop(['insurance_benefits','insurance_benefits_received'], axis=1) # extrae las características
    target = data['insurance_benefits_received'] # extrae los objetivos
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=12345)
    
    f1_scr = []
    for k in range(1,neighbors):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(features_train, target_train)
        knn_predict = knn_model.predict(features_test)
    
        f1_score = sklearn.metrics.f1_score(target_test, knn_predict)
        f1_scr.append(f1_score)
        
    return f1_scr


# In[28]:


get_knn_clss_f1 = get_knn_class(df, 11)
get_knn_clss_f1


# In[29]:


get_knn_clss_f1 = get_knn_class(df_scaled, 11)
get_knn_clss_f1


# In[30]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo    
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)


# In[31]:


# generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[32]:


# Probabilidades del modelo

for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'La probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, df['insurance_benefits_received'].shape[0], seed=42)
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# Comentarios Tarea 2.
# 
# * Cálculo del objetivo (en datos originales y escalados)
# 
# * Presentación del desequilibrio de clases
# 
# * Definición y aplicación de funciones:
#     * get_knn_class: Las evaluaciones F1 para los datos escalados son por mucho mejores que las de los datos originales.
#     * eval_classifier
#     * rnd_model_predict
#     
# * Probabilidades del modelo
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Destacas de manera clara cómo el escalado de los datos influye significativamente en el desempeño del clasificador, mostrando mejoras notables en la métrica F1. También se aprecia tu atención al desequilibrio de clases, un aspecto crucial en problemas de clasificación que puede afectar fuertemente los resultados y su interpretación. La aplicación de funciones personalizadas aporta valor al análisis, permitiendo una evaluación más precisa del modelo.
# 
# </div>

# -----------------

# # Tarea 3. Regresión (con regresión lineal)

# Con `insurance_benefits` como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

# Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?
# 
# Denotemos
# 
# - $X$: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades
# - $y$ — objetivo (un vector)
# - $\hat{y}$ — objetivo estimado (un vector)
# - $w$ — vector de pesos
# 
# La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
# 
# $$
# y = Xw
# $$
# 
# El objetivo de entrenamiento es entonces encontrar esa $w$ w que minimice la distancia L2 (ECM) entre $Xw$ y $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# Parece que hay una solución analítica para lo anteriormente expuesto:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# La fórmula anterior puede servir para encontrar los pesos $w$ y estos últimos pueden utilizarse para calcular los valores predichos
# 
# $$
# \hat{y} = X_{val}w
# $$

# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

# In[33]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        self.w = self.weights[1:]
        self.w0 = self.weights[0]

    def predict(self, X):
        
        # añadir las unidades
        #X2 = # <tu código aquí>
        y_pred = X.dot(self.w) + self.w0
        
        return y_pred


# In[34]:


def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# In[35]:


# Resultados del modelo (pesos, RMSE y R2)

X = df[['gender', 'age', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print("Pesos:")
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# In[36]:


# Resultados del modelo con datos escalados (pesos, RMSE y R2)

X = df_scaled[['gender', 'age', 'income', 'family_members']].to_numpy()
y = df_scaled['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print("Pesos:")
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# Comentarios Tarea 3.
# 
# * Creación y aplicación de clase:
#     * MyLinearRegression
#     
# * Definición y aplicación de funciones:
#     * eval_regressor
#     
# * Resultados del modelo con datos originales y escalados (pesos, RMSE y R2)
# 
# ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?
# 
# * Las métricas de evaluación son similares sin embargo los pesos se adaptan a sus magnitudes en ambos casos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteración)</b> <a class=“tocSkip”></a>
#     
# Muy buen implementación de las funciones para el modelo de regresión lineal. Se evidencia que este caso tiene una tendencia más hacia ser un problema de clasificación para `insurance_benefits`, ya que cuando realizamos el análisis como una regresión el rendimiento de este modelo no es tan bueno como se puede observar con el **R2**
#     
# 
# </div>

# ---------------

# # Tarea 4. Ofuscar datos

# Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.
# 
# Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

# In[37]:


# Determinación de matriz
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[38]:


X = df_pn.to_numpy()
X


# Generar una matriz aleatoria $P$.

# In[39]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))
P


# Comprobar que la matriz P sea invertible

# In[40]:


P_inv = np.linalg.inv(P)
P_inv


# In[41]:


X_1 = np.dot(X, P)
X_1
len(X_1)


# ¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

# No se pueden saber los datos ya que quedan afectados por operaciones con números aleatorios.

# ¿Puedes recuperar los datos originales de $X'$ si conoces $P$? Intenta comprobarlo a través de los cálculos moviendo $P$ del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

# In[42]:


X_o = np.dot(X_1, P_inv)
X_o


# Sí es posible recuperar los datos originales.

# Muestra los tres casos para algunos clientes
# 
# - Datos originales
# - El que está transformado
# - El que está invertido (recuperado)

# In[43]:


print("Datos originales")
print(X[0:5])
print()
print("Datos transformados")
print(X_1[0:5])
print()
print("inversión de datos transformados")
print(X_o[0:5])


# Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

# Las operaciones decimales.

# In[44]:


pd.options.display.float_format = '{:.2f}'.format
print()
print("Datos originales")
df_X = pd.DataFrame(X[0:5], columns=personal_info_column_list)
print(df_X)
print()
print()
print("Datos transformados")
df_X_1 = pd.DataFrame(X_1[0:5], columns=personal_info_column_list)
print(df_X_1)
print()
print()
print("inversión de datos transformados")
df_X_o = pd.DataFrame(X_o[0:5], columns=personal_info_column_list)
print(df_X_o)


# Comentarios Tarea 4.
# 
# * Determinación de matriz
# * Generación de matriz aleatoria
# * Comprobación de invertibilidad
# * Obtención de Matriz prima
# * Inversión de Matriz prima

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# En ocasiones los valores se pueden ver ligeramente alterados, sobretodo si son categóricos. Habría que hacer una revisión si es verdad que todos los valores que originalmente eran 0 ahora son negativos. Aunque lo que debes notar es que estos valores aparecen acompañado de un expenente `e-12` o `e-13` lo que quiere decir que es prácticamente 0
# </div>

# ---------------

# ## Prueba de que la ofuscación de datos puede funcionar con regresión lineal

# En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar _analytically_ que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

# Entonces, los datos están ofuscados y ahora tenemos $X \times P$ en lugar de tener solo $X$. En consecuencia, hay otros pesos $w_P$ como
# 
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# ¿Cómo se relacionarían $w$ y $w_P$ si simplificáramos la fórmula de $w_P$ anterior?

# **Respuesta**

# Deberían tener el mismo impacto en magnitudes diferentes, ya que $w_P$ se ve afectado por la matriz aleatoria $P$

# ¿Cuáles serían los valores predichos con $w_P$? 
# 

# **Respuesta**

# Tendrían un efecto similar a los datos escalados.

# ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?

# **Respuesta**

# La métrica de RECM debería ser la misma para los datos originales y ofuscados.

# Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!

# 
# No es necesario escribir código en esta sección, basta con una explicación analítica.

# **Prueba analítica**

# $$
# w_P = I P^{-1} X^{-1} y
# $$

# ----------

# ## Prueba de regresión lineal con ofuscación de datos

# Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
# 
# Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación.
# 
# Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y $R^2$. ¿Hay alguna diferencia?

# **Procedimiento**
# 
# - Crea una matriz cuadrada $P$ de números aleatorios.
# - Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.
# - Se utilizó la matriz aleatoria previamente generadas.
# - Utiliza $XP$ como la nueva matriz de características.

# In[45]:


# Resultados del modelo (pesos, RMSE y R2)

X = df[['gender', 'age', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_1_train, X_1_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=12345)

lr_o = MyLinearRegression()

lr_o.fit(X_1_train, y_train)
print("Pesos:")
print(lr.weights)

y_test_pred_o = lr_o.predict(X_1_test)
eval_regressor(y_test, y_test_pred_o)


# Comentarios Tarea 5.
# 
# * Resultados del modelo (pesos, RMSE y R2): Las métricas obtenidas en el modelo con datos ofuscados se mantienen en los mismos valores de los datos originales y escalados, los pesos se adaptan a las magnitudes de cada conjunto sin embargo los pesos de los datos ofuscados y los escalados son prácticamente los mismos.
# 
# Datos originales
# 
# Pesos:
# [-9.43539012e-01,  1.64272726e-02,  3.57495491e-02, -2.60743659e-07, -1.16902127e-02]
# 
# RMSE: 0.34
# 
# R2: 0.66
# 
# Datos escalados
# 
# Pesos:
# [-0.94353901,  0.01642727,  2.32372069, -0.02059875, -0.07014128]
# 
# RMSE: 0.34
# 
# R2: 0.66

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteración)</b> <a class=“tocSkip”></a>
#     
# Oscar en esta sección has realizado el proceso de ofuscación de datos de manera correcta. Y vemos como el modelo no se ve afectado con respecto a la prueba con los datos originales. Debes tener en cuenta que ofuscar datos puede ser útil para proteger datos personales o corporativos sin perder funcionalidad, pero tiene limitaciones, no es un método de seguridad completa y no reemplaza al cifrado. Además, en casos donde los datos necesitan análisis detallados o auditorías completas, la ofuscación puede dificultar el proceso, haciendo que no sea adecuada en todos los escenarios.
# 
# </div>

# ----------

# # Conclusiones

# Conclusiones.
# 
# * Se pueden apreciar los efectos que genera el escalado de datos, también la ofuscación así como revertir el efecto de la ofuscación y comparar los resultados de los pesos y las métricas a manera de saber que se llegan a los mismos resultados en el caso de las métricas.
# 
# * Cabe señalar que el enfoque del proyecto no fue necesariamente la mejora de los modelos sino las alternativas al trabajar con matrices.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteración)</b> <a class=“tocSkip”></a>
# 
# Oscar excelente conclusión, ésta refleja una observación precisa y bien fundamentada sobre la consistencia de las métricas a pesar de las transformaciones aplicadas a los datos, lo cual demuestra una comprensión sólida de cómo el escalado y la ofuscación afectan principalmente a los coeficientes sin alterar el rendimiento del modelo. Además, es acertado cómo señalas que el objetivo del proyecto no era optimizar el modelo, sino explorar distintas formas de manipular datos manteniendo su integridad.
# 
# Para enriquecer futuros análisis, podrías explorar visualizaciones que evidencien estos efectos en los pesos o bien discutir en qué contextos prácticos la ofuscación de datos puede ser útil, especialmente en entornos donde la privacidad es crítica.
# </div>

# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter.

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores
# - [ ]  Las celdas están ordenadas de acuerdo con la lógica y el orden de ejecución
# - [ ]  Se ha realizado la tarea 1
# - [ ]  Está presente el procedimiento que puede devolver k clientes similares para un cliente determinado
# - [ ]  Se probó el procedimiento para las cuatro combinaciones propuestas    
# - [ ]  Se respondieron las preguntas sobre la escala/distancia
# - [ ]  Se ha realizado la tarea 2
# - [ ]  Se construyó y probó el modelo de clasificación aleatoria para todos los niveles de probabilidad    
# - [ ]  Se construyó y probó el modelo de clasificación kNN tanto para los datos originales como para los escalados. Se calculó la métrica F1.
# - [ ]  Se ha realizado la tarea 3
# - [ ]  Se implementó la solución de regresión lineal mediante operaciones matriciales    
# - [ ]  Se calculó la RECM para la solución implementada
# - [ ]  Se ha realizado la tarea 4
# - [ ]  Se ofuscaron los datos mediante una matriz aleatoria e invertible P    
# - [ ]  Se recuperaron los datos ofuscados y se han mostrado algunos ejemplos    
# - [ ]  Se proporcionó la prueba analítica de que la transformación no afecta a la RECM    
# - [ ]  Se proporcionó la prueba computacional de que la transformación no afecta a la RECM
# - [ ]  Se han sacado conclusiones

# # Apéndices
# 
# ## Apéndice A: Escribir fórmulas en los cuadernos de Jupyter

# Puedes escribir fórmulas en tu Jupyter Notebook utilizando un lenguaje de marcado proporcionado por un sistema de publicación de alta calidad llamado $\LaTeX$ (se pronuncia como "Lah-tech"). Las fórmulas se verán como las de los libros de texto.
# 
# Para incorporar una fórmula a un texto, pon el signo de dólar (\\$) antes y después del texto de la fórmula, por ejemplo: $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# Si una fórmula debe estar en el mismo párrafo, pon el doble signo de dólar (\\$\\$) antes y después del texto de la fórmula, por ejemplo:
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# El lenguaje de marcado de [LaTeX](https://es.wikipedia.org/wiki/LaTeX) es muy popular entre las personas que utilizan fórmulas en sus artículos, libros y textos. Puede resultar complicado, pero sus fundamentos son sencillos. Consulta esta [ficha de ayuda](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) (materiales en inglés) de dos páginas para aprender a componer las fórmulas más comunes.

# ## Apéndice B: Propiedades de las matrices

# Las matrices tienen muchas propiedades en cuanto al álgebra lineal. Aquí se enumeran algunas de ellas que pueden ayudarte a la hora de realizar la prueba analítica de este proyecto.

# <table>
# <tr>
# <td>Distributividad</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>No conmutatividad</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Propiedad asociativa de la multiplicación</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Propiedad de identidad multiplicativa</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversibilidad de la transposición de un producto de matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>
