# %% [markdown]
# # División de los datos en grupo para entrenamiento, prueba y validación

# %%
import pandas as pd 
import numpy as np 

df = pd.read_excel("Datos IRIS.xlsx", index_col=None, header=None)

# División de dataset

df_T = df.iloc[0:35] # Datos de Entrenamiento
df_T = df_T.append(df.iloc[50:85])
df_T = df_T.append(df.iloc[100:135])

print("Entrenamiento \n", df_T)


df_P = df.iloc[35:45]          #Datos de prueba
df_P = df_P.append(df.iloc[85:95])
df_P = df_P.append(df.iloc[135:145])

print("Prueba \n", df_P)

df_R = df.iloc[45:50]          #Datos de validación
df_R = df_R.append(df.iloc[95:100])
df_R = df_R.append(df.iloc[145:150])

print("Resultantes \n", df_R)



# %% [markdown]
# # Normalización de los conjutos de datos

# %%
#Mínimos y máximos
def minimo(dfin):
    min_0 = dfin[0].min()  #Columna 0
    min_1 = dfin[1].min()  #Columna 1
    min_2 = dfin[2].min()  #Columna 2
    min_3 = dfin[3].min()  #Columna 3

    return [min_0, min_1, min_2, min_3]

def maximo(dfin):
    max_0 = dfin[0].max()  #Columna 0
    max_1 = dfin[1].max()  #Columna 1
    max_2 = dfin[2].max()  #Columna 2
    max_3 = dfin[3].max()  #Columna 3

    return [max_0, max_1, max_2, max_3]

# Normalización por columna
def minmax_norm(dfin , min, max):
    return (dfin - min) / (max - min)

# Normalización del dataframe
def norm_df(dfin, min = [0,0,0,0], max = [0,0,0,0]):

    # Normalización de Dataframe de entrenamiento
    df_norm_0 = minmax_norm(dfin[0], min[0], max[0])  #Columna 0
    df_norm_1 = minmax_norm(dfin[1], min[1], max[1])  #Columna 1
    df_norm_2 = minmax_norm(dfin[2], min[2], max[2])  #Columna 2
    df_norm_3 = minmax_norm(dfin[3], min[3], max[3])  #Columna 3

    return pd.DataFrame([df_norm_0,df_norm_1, df_norm_2, df_norm_3, dfin[5]] ).transpose()  # Dataframe con las columnas normalizadas

#Mínimos y máximos para cada Dataframe
global_min = minimo(df_T)    # Mínimo global de los datos
global_max = maximo(df_T)    # Máximo global de los datos

minimo_P = minimo(df_P)
maximo_P = maximo(df_P)

minimo_R = minimo(df_R)
maximo_R = maximo(df_R)

#Actualización de mínimo y máximo global en caso de que sea necesario
for i in range(len(minimo_P)):
    if minimo_P[i] < global_min[i]:
        global_min[i] = minimo_P[i]
    if maximo_P[i] > global_max[i]:
        global_max[i] = maximo_P[i]

for i in range(len(minimo_R)):
    if minimo_R[i] < global_min[i]:
        global_min[i] = minimo_R[i]
    if maximo_R[i] > global_max[i]:
        global_max[i] = maximo_R[i]

print(global_min, global_max)

#Normalización Dataframe de entrenamiento
df_TN = norm_df(df_T, global_min, global_max) 
print("Entrenamiento Normalizado \n", df_TN)

#Normalización Dataframe de prueba
df_PN = norm_df(df_P, global_min, global_max)
print("Prueba Normalizado \n",df_PN)

# Normalización de Dataframe de validación
df_norm_0 = minmax_norm(df_R[0], global_min[0], global_max[0])  #Columna 0
df_norm_1 = minmax_norm(df_R[1], global_min[1], global_max[1])  #Columna 1
df_norm_2 = minmax_norm(df_R[2], global_min[2], global_max[2])  #Columna 2
df_norm_3 = minmax_norm(df_R[3], global_min[3], global_max[3])  #Columna 3

df_RN = pd.DataFrame([df_norm_0,df_norm_1, df_norm_2, df_norm_3]).transpose()  # Dataframe con las columnas normalizadas

print("Validación Normalizado \n",df_RN)

# %% [markdown]
# # Descriptores

# %%
import matplotlib.pyplot as plt

my_plot = df_TN.plot(1, 0, kind="scatter")
plt.show()

my_plot = df_TN.plot(2, 0, kind="scatter")
plt.show()

my_plot = df_TN.plot(3, 0, kind="scatter")
plt.show()

my_plot = df_TN.plot(2, 1, kind="scatter")
plt.show()

my_plot = df_TN.plot(3, 1, kind="scatter")
plt.show()

my_plot = df_TN.plot(3, 2, kind="scatter")
plt.show()

# %%
my_plot = df_TN.plot(0, 5, kind="scatter", xlabel= "X0", ylabel = "Y1")
plt.show()
my_plot = df_TN.plot(1, 5, kind="scatter", xlabel= "X1", ylabel = "Y1")
plt.show()
my_plot = df_TN.plot(2, 5, kind="scatter", xlabel= "X2", ylabel = "Y1")
plt.show()
my_plot = df_TN.plot(3, 5, kind="scatter", xlabel= "X3", ylabel = "Y1")
plt.show()

# %%
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(df_TN[1], df_TN[5], df_TN[3], c='g', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Clase')
ax1.set_zlabel('Z')
ax1.legend()

plt.show()

# %% [markdown]
# # MONOCAPA

# %%
import random
Tolerancia = 0.03 # Valor de tolerancia de error
error = 100       # Error grande

error_T = []  # Arreglo para almacenar los valores del error para cada elemento de entrenamiento
error_P = []  # Arreglo para almacenar los valores del error para cada elemento de prueba
w0 = [1,1,0]  # Pesos iniciales arbitrarios
w0 = np.array(w0) 
iter = 0      # Contador de iteraciones
u = 0.04      # Paso para modificación de pesos
t = 0         # Cantidad de aciertos continuos para mejorar el algoritmo

while (np.abs(error) > Tolerancia or t < 4):  # Cuando el error sea menor a la toleracia o los aciertos continuos son menores a 4
    ########## Entrenamiento #########
    i = random.randint(0, 104)  # Índice aleatorio
    x1,x2,y =  df_TN.iloc[i, 2], df_TN.iloc[i, 3], df_TN.iloc[i, 4]  # Entradas y valor esperado
    x = np.array([x1,x2,1])  # Arreglo de entradas
    x_w = (w0.T * x)         # Multiplicación de entradas por los pesos
    suma = x_w[0] + x_w[1] + x_w[2] # Sumatoria de los productos
    error = suma - y                # Cálculo del error
    error_T.append(error)           # Se agrega el error calculado al arreglo de errores de entrenamiento

    ###########    Prueba  ##########
    j = random.randint(0, 29) # Índice aleatorio
    x3,x4,y1 =  df_PN.iloc[j, 2], df_PN.iloc[j, 3], df_PN.iloc[j, 4]  # Entradas y valor esperado
    x_P = np.array([x3,x4,1])   # Arreglo de entradas
    xP_w = (w0.T * x_P)         # Multiplicación de entradas por los pesos
    sumaP = xP_w[0] + xP_w[1] + xP_w[2]  # Sumatoria de los productos
    errorP = sumaP - y1         # Cálculo del error
    error_P.append(errorP)      # Se agrega el error calculado al arreglo de errores de prueba

    if  np.abs(error) > Tolerancia:   # Condición para modificar los pesos de la siguiente iteración
        w0 = w0 - u*x*error           # Modificación de los pesos
        t = 0                         # Reinicia contador de aciertos
    elif  np.abs(error) < Tolerancia: # Condición de acierto
        t += 1                        #Contar acierto

    iter += 1      # Contador de iteraciones

    # Condiciones para salir del ciclo while
    if iter > 2000:
        error = 0.06
        t = 6

####### Gráfica de resultados ################
x = np.arange(0,len(error_T))    # Vector que representa el número de iteraciones

plt.plot(x, np.abs(error_T), 'x', label = "Entrenamiento")  #Gráfica de arreglo de errores de Entrenamiento
plt.plot(x, np.abs(error_P), 'o',label = "Prueba")          #Gráfica de arreglo de errores de Prueba
plt.xlabel("Iteraciones")    # Etiqueta de eje X
plt.ylabel("Error")          # Etiqueta de eje Y
plt.title("Error Entrenamiento y Prueba")   # Título del gráfico
plt.legend()
plt.show()

# %%
error_R = [] #Arreglo de resultados obtenidos 
for k in range(15):
    
    ##########   Validación   ###############
    x3,x4 =  df_RN.iloc[k, 2], df_RN.iloc[k, 3] # Entradas
    x_R = np.array([x3,x4,1])                   # Arreglo de entradas
    xR_w = (w0.T * x_R)                         # Multiplicación de entradas por los pesos
    sumaR = xR_w[0] + xR_w[1] + xR_w[2]         # Sumatoria de los productos
    error_R.append(sumaR)                       # Se agrega el resultado de la sumatoria

####### Gráfica de resultados #########
x = np.arange(0,len(error_R)) # Vector que representa el número de elementos

plt.plot(x, np.abs(error_R), 'o') #Gráfica de arreglo de resultado obtenido para cada elemento
plt.xlabel("Elemento")
plt.ylabel("Valor obtenido")
plt.title("Validación")
plt.show()

# %% [markdown]
# # Multicapa

# %%
#Función sigmoide
def sig(x):
    h = 1 / (1 + np.exp(x))
    return h

Tolerancia = 0  # Valor de tolerancia de error
error = 100        # Error grande
errorP = 100       # Error Prueba grande
error_salida = 100 # # Error dela salida de la red

error_T = []       # Arreglo para almacenar los valores del error para cada elemento de entrenamiento
error_P = []       # Arreglo para almacenar los valores del error para cada elemento de prueba
error_s = []       # Arreglo para almacenar los valores del error para cada salida
w0 = np.array([1, 1, 0, 1, 1,0])  # Pesos para primera capa, con bias 
w1 = np.array([0.5,0.2, 0])       # Pesos para la segunda capa, con bias

w0_post = np.array([0,0,0,0,0,0]) # Pesos para primera capa siguiente iteración, con bias
w1_post = np.array([0,0,0])       # Pesos para la segunda capa siguiente iteración, con bias

iter = 0                #Contador de iteraciones
t = 0                   #Contador de aciertos continuos

u = 0.002               #Paso para modificación de pesos

while (np.abs(error) != Tolerancia) or t < 100:   # Cuando el error sea menor a la toleracia o los aciertos continuos son menores a 30
    ########### Entrenamiento  #############
    i = random.randint(0, 104)  # Índice aleatorio
    x1,x2,y =  df_TN.iloc[i, 2], df_TN.iloc[i, 3], df_TN.iloc[i, 4]   # Entradas y valor esperado
    x = np.array([x1,x2,1])     # Arreglo de entradas

    ###########    Prueba    ###############
    j = random.randint(0, 29)  # Índice aleatorio
    x3,x4,y1 =  df_PN.iloc[j, 2], df_PN.iloc[j, 3], df_PN.iloc[j, 4] # Entradas y valor esperado
    x_P = np.array([x3,x4,1])    # Arreglo de entradas
    xP_w0 = (w0[0:3].T * x_P)    # Multiplicación de entradas con los pesos para la neurona 1
    xP_w1 = (w0[3:6].T * x_P)    # Multiplicación de entradas con los pesos para la neurona 2
    sumP0 = xP_w0[0] + xP_w0[1] + xP_w0[2]  #Sumatoria de productos para neurona 1
    sumP1 = xP_w1[0] + xP_w1[1] + xP_w1[2]  #Sumatoria de productos para neurona 2
    h1P = sig(sumP0)             # Función de transferencia para resultado de la suma de la neurona 1
    h2P = sig(sumP1)             # Función de transferencia para resultado de la suma de la neurona 2
    h_P = np.array([h1P,h2P,1])  # Arreglo de resultado de las neuronas 1 y 2
    xP_w2 = (w1.T * h_P)         # Multiplicación de arreglo de resultados de la capa interna por los pesos de la segunda capa
    sumP2 = xP_w2[0] + xP_w2[1] + xP_w2[2] #Sumatoria de los productos para la segunda capa
    resultP = sig(sumP2)         # Resultado final

    error_salida_P = resultP * (1 - resultP) * (y - resultP)  # Error del resultado obtenido
    if np.abs(error_salida_P) < 0.3:
        errorP = 1 - y
    elif np.abs(error_salida_P) < 0.39:
        errorP = 2 - y
    elif np.abs(error_salida_P) < 0.49:
        errorP = 3 - y
    error_P.append(errorP)

    #Entradas multiplicadas por los pesos de la primera capa
    x_w0 = (w0[0:3].T * x)
    x_w1 = (w0[3:6].T * x)

    # Suma de la multiplicación de cada entrada de las neuronas
    sum0 = x_w0[0] + x_w0[1] + x_w0[2]
    sum1 = x_w1[0] + x_w1[1] + x_w1[2]

    # Función de transferencia de cada neurona
    h1 = sig(sum0)
    h2 = sig(sum1)

    h = np.array([h1,h2,1])

    ##### Segunda capa  ###
    x_w2 = (w1.T * h)
    sum2 = x_w2[0] + x_w2[1] + x_w2[2]
    result = sig(sum2)

    # Cálculo de error de la salida
    error_salida = result * (1 - result) * (y - result)
    error_s.append(error_salida)

    if np.abs(error_salida) > Tolerancia:

        # Cálculo de los pesos de la capa 2
        w1_post[0] = np.round(w1[0] + u*h1*error_salida)
        w1_post[1] = np.round(w1[0] + u*h2*error_salida)

        ##### Cálculo de error para la capa 1 #######
        # Neurona 1
        error_N1 = h1*(1-h1)*(w1[0]*error_salida)

        # Neurona 2
        error_N2 = h2*(1-h2)*(w1[1]*error_salida)

        # Cálculo de los pesos de la capa 1 para la neurona 1
        w0_post[0] = w0[0] + u*x1*error_N1
        w0_post[1] = w0[1] + u*x2*error_N1

        # Cálculo de los pesos de la capa 1 para la neurona 2
        w0_post[3] = w0[3] + u*x1*error_N2
        w0_post[4] = w0[4] + u*x2*error_N2

        ##### actualización de los pesos de las 2 capas para la siguiente iteración
        w0 = w0_post
        w1 = w1_post
        
    if np.abs(result) < 0.3:  # Resultado para clase 1
        error = 1 - y
    elif np.abs(result) < 0.4: # Resultado para clase 2
        error = 2- y
    elif np.abs(result) < 0.5: # Resultado para clase 3
        error = 3 - y
    error_T.append(error)

    iter += 1        # Contador de iteraciones
    if iter > 200:   # Condición de iteraciones para salir del ciclo while
        error_salida = 0.06
        t = 200
    elif error == 0:  # Contador de aciertos continuos
        t += 1
    else:
        t = 0        #Reinicio del contador de aciertos continuos


# %%
x = np.arange(0,len(error_T))  # Vector de iteraciones 

################### Gráfica ##################
plt.plot(x, np.abs(error_T), 'x', label = "Entrenamiento")
plt.plot(x, np.abs(error_P), 'o', label = "Prueba")
plt.xlabel("Iteraciones")    # Etiqueta de eje X
plt.ylabel("Error")          # Etiqueta de eje Y
plt.title("Error Entrenamiento y Prueba")   # Título del gráfico
plt.legend()
plt.show()

plt.plot(x, np.abs(error_s), 'o')
plt.xlabel("Iteraciones")    # Etiqueta de eje X
plt.ylabel("Diferencia en la salida")          # Etiqueta de eje Y
plt.title("Diferencia según la clase")   # Título del gráfico
plt.show()

# %%
error_R = []  #Arreglo de resultados obtenidos
for k in range(15):

    ##########   Validación   ###############
    x3,x4 =  df_RN.iloc[k, 2], df_RN.iloc[k, 3]  # Entradas
    x_R = np.array([x3, x4, 1])                  # Arreglo de entradas      
    xR_w0 = (w0[0:3].T * x_R)                    # Multiplicación de entradas por los pesos de la neurona 1
    xR_w1 = (w0[3:6].T * x_R)                    # Multiplicación de entradas por los pesos de la neurona 2
    sumR0 = xR_w0[0] + xR_w0[1] + xR_w0[2]       # Sumatoria de los productos de la neurona 1
    sumR1 = xR_w1[0] + xR_w1[1] + xR_w1[2]       # Sumatoria de los productos de la neurona 2
    h1R = sig(sumR0)                             # Función de transferencia de cada neurona
    h2R = sig(sumR1)
    h_R = np.array([h1R,h2R,1])                  # Arreglo de resultado de las neuronas 1 y 2
    xR_w2 = (w1.T * h_R)                         # Multiplicación de arreglo de resultados de la capa interna por los pesos de la segunda capa
    sumR2 = xR_w2[0] + xR_w2[1] + xR_w2[2]       #Sumatoria de los productos para la segunda capa
    resultR = sig(sumR2)                         # Resultado final
    if np.abs(resultR ) < 0.3:
        error_R.append(1)
    elif np.abs(resultR ) < 0.4:
        error_R.append(2)
    elif np.abs(resultR ) < 0.5:
        error_R.append(3)

x = np.arange(0,len(error_R))

plt.plot(x, np.abs(error_R), 'o')
plt.xlabel("Elemento")
plt.ylabel("Valor obtenido")
plt.title("Validación")
plt.show()

# %% [markdown]
# # Monocapa con 4 variables

# %%
Tolerancia = 0.05  # Valor de tolerancia de error
error = 100        # Error grande

error_T = []       # Arreglo para almacenar los valores del error para cada elemento de entrenamiento
error_P = []       # Arreglo para almacenar los valores del error para cada elemento de prueba
w0 = [1,1,1,1,0]   # Pesos iniciales arbitrarios
w0 = np.array(w0)
iter = 0           # Contador de iteraciones
u = 0.04           # Paso para modificación de pesos
t = 0              # Cantidad de aciertos continuos para mejorar el algoritmo

while (np.abs(error) > Tolerancia or t < 4):  # Cuando el error sea menor a la toleracia o los aciertos continuos son menores a 4
    ########## Entrenamiento #########
    i = random.randint(0, 104) # Índice aleatorio
    x0,x1,x2,x3,y =  df_TN.iloc[i, 0], df_TN.iloc[i, 1], df_TN.iloc[i, 2], df_TN.iloc[i, 3], df_TN.iloc[i, 4] # Entradas y valor esperado
    x = np.array([x0,x1,x2,x3,1])                        # Arreglo de entradas
    x_w = (w0.T * x)                                     # Multiplicación de entradas por los pesos
    suma = x_w[0] + x_w[1] + x_w[2] + x_w[3] +x_w[4]     # Sumatoria de los productos
    error = suma - y                                     # Cálculo del error
    error_T.append(error)                                # Se agrega el error calculado al arreglo de errores de entrenamiento

     ###########    Prueba  ##########
    j = random.randint(0, 29)      # Índice aleatorio
    x4,x5,x6,x7,y1 =  df_PN.iloc[j, 0],df_PN.iloc[j, 1],df_PN.iloc[j, 2], df_PN.iloc[j, 3], df_PN.iloc[j, 4] # Entradas y valor esperado
    x_P = np.array([x4,x5,x6,x7,1])                      # Arreglo de entradas
    xP_w = (w0.T * x_P)                                  # Multiplicación de entradas por los pesos
    sumaP = xP_w[0] + xP_w[1] + xP_w[2] + xP_w[3] + xP_w[4]   # Sumatoria de los productos
    errorP = sumaP - y1                                  # Cálculo del error
    error_P.append(errorP)                               # Se agrega el error calculado al arreglo de errores de prueba

    if  np.abs(error) > Tolerancia:
        w0 = w0 - u*x*error
        t = 0
    elif  np.abs(error) < Tolerancia:
        t += 1

    iter += 1
    if iter > 5000:
        error = 0.02
        t = 6
    
####### Gráfica de resultados ################
x = np.arange(0,len(error_T))    # Vector que representa el número de iteraciones

plt.plot(x, np.abs(error_T), 'x', label = "Entrenamiento")  #Gráfica de arreglo de errores de Entrenamiento
plt.plot(x, np.abs(error_P), 'o',label = "Prueba")          #Gráfica de arreglo de errores de Prueba
plt.xlabel("Iteraciones")    # Etiqueta de eje X
plt.ylabel("Error")          # Etiqueta de eje Y
plt.title("Error Entrenamiento y Prueba")   # Título del gráfico
plt.legend()
plt.show()

# %%
error_R = []
for k in range(15):

    ########## Validación  ############
    x1,x2,x3,x4 =  df_RN.iloc[k, 0], df_RN.iloc[k, 1],df_RN.iloc[k, 2], df_RN.iloc[k, 3]  # Entradas
    x_R = np.array([x1,x2,x3,x4,1])                                                       # Arreglo de entradas
    xR_w = (w0.T * x_R)                                                                   # Multiplicación de entradas por los pesos
    sumaR = xR_w[0] + xR_w[1] + xR_w[2] + xR_w[3] +xR_w[4]                                # Sumatoria de los productos
    error_R.append(sumaR)                                                                 # Se agrega el resultado calculado al arreglo error_R


x = np.arange(0,len(error_R))    # Vector que representa el número de iteraciones
plt.plot(x, np.abs(error_R), 'o')

plt.xlabel("Elemento")
plt.ylabel("Valor obtenido")
plt.title("Validación 4 descriptores")
plt.show()


