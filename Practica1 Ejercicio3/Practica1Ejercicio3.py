import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir la función de activación (función sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de pesos y sesgos para la red neuronal
def inicializar_pesos(layers):
    pesos = []
    sesgos = []
    for i in range(1, len(layers)):
        matriz_pesos = np.random.rand(layers[i - 1], layers[i])
        sesgo = np.zeros((1, layers[i]))
        pesos.append(matriz_pesos)
        sesgos.append(sesgo)
    return pesos, sesgos

# Propagación hacia adelante en la red neuronal
def propagacion_hacia_adelante(X, pesos, sesgos):
    activaciones = [X]
    for i in range(len(pesos)):
        entrada_neta = np.dot(activaciones[-1], pesos[i]) + sesgos[i]
        activacion = sigmoid(entrada_neta)
        activaciones.append(activacion)
    return activaciones

# Retropropagación en la red neuronal (gradiente descendente)
def retropropagacion(X, y, activaciones, pesos, sesgos, tasa_aprendizaje):
    num_capas = len(pesos)
    errores = [y - activaciones[-1]]
    deltas = [errores[-1] * sigmoid_derivative(activaciones[-1])]

    # Calcular errores y deltas para capas ocultas
    for i in range(num_capas - 2, -1, -1):
        error = deltas[-1].dot(pesos[i + 1].T)
        delta = error * sigmoid_derivative(activaciones[i + 1])
        errores.append(error)
        deltas.append(delta)

    # Invertir deltas para que estén en el orden correcto
    deltas = deltas[::-1]

    # Actualizar pesos y sesgos usando gradientes y deltas
    for i in range(num_capas - 1, -1, -1):
        pesos[i] += tasa_aprendizaje * activaciones[i].T.dot(deltas[i])
        sesgos[i] += tasa_aprendizaje * np.sum(deltas[i], axis=0, keepdims=True)

    # Calcular el error total
    error_total = np.mean(np.abs(errores[-1]))
    return pesos, sesgos, error_total

# Entrenamiento del perceptrón multicapa
def entrenar_perceptron_multicapa(X, y, capas, tasa_aprendizaje, epocas):
    num_muestras, num_caracteristicas = X.shape
    pesos, sesgos = inicializar_pesos(capas)

    for _ in range(epocas):
        for i in range(num_muestras):
            datos_entrada = X[i].reshape(1, -1)
            salida_objetivo = y[i].reshape(1, -1)

            activaciones = propagacion_hacia_adelante(datos_entrada, pesos, sesgos)
            pesos, sesgos, _ = retropropagacion(datos_entrada, salida_objetivo, activaciones, pesos, sesgos, tasa_aprendizaje)

    return pesos, sesgos

# Clasificación usando el perceptrón multicapa entrenado
def predecir_perceptron_multicapa(X, pesos, sesgos):
    activaciones = propagacion_hacia_adelante(X, pesos, sesgos)
    predicciones = (activaciones[-1] > 0.5).astype(int)
    return predicciones

# Lectura de datos desde el archivo CSV
def cargar_datos(nombre_archivo):
    datos = pd.read_csv(nombre_archivo)
    X = datos.iloc[:, :-1].values
    y = datos.iloc[:, -1].values.reshape(-1, 1)
    return X, y

# Visualización del resultado de la clasificación
def visualizar_clasificacion(X, y, predicciones):
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=predicciones.flatten(), marker='x', cmap='coolwarm', s=100, linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clasificación de Perceptrón Multicapa')
    plt.show()

# Parámetros de la red neuronal
tamano_entrada = 2  # Cantidad de características de entrada
tamano_salida = 1  # Cantidad de neuronas en la capa de salida
capas_ocultas = int(input("Ingrese la cantidad de capas ocultas: "))
neuronas_por_capa_oculta = int(input("Ingrese la cantidad de neuronas por capa oculta: "))
tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje: "))
epocas = int(input("Ingrese el número de épocas: "))

# Cargar y preparar los datos
X, y = cargar_datos('concentlite.csv')

# Definir la arquitectura de la red neuronal
capas = [tamano_entrada] + [neuronas_por_capa_oculta] * capas_ocultas + [tamano_salida]

# Entrenar el perceptrón multicapa
pesos_entrenados, sesgos_entrenados = entrenar_perceptron_multicapa(X, y, capas, tasa_aprendizaje, epocas)

# Realizar predicciones
predicciones = predecir_perceptron_multicapa(X, pesos_entrenados, sesgos_entrenados)

# Visualizar resultados
visualizar_clasificacion(X, y, predicciones)
