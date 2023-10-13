
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define la función de activación (función sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de pesos y sesgos para la red neuronal
def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(1, len(layers)):
        weight_matrix = np.random.rand(layers[i - 1], layers[i])
        bias = np.zeros((1, layers[i]))
        weights.append(weight_matrix)
        biases.append(bias)
    return weights, biases

# Propagación hacia adelante en la red neuronal
def feed_forward(X, weights, biases):
    activations = [X]
    for i in range(len(weights)):
        weighted_input = np.dot(activations[-1], weights[i]) + biases[i]
        activation = sigmoid(weighted_input)
        activations.append(activation)
    return activations

# Retropropagación en la red neuronal (gradiente descendente)
def backpropagation(X, y, activations, weights, biases, learning_rate):
    num_layers = len(weights)
    errors = [y - activations[-1]]
    deltas = [errors[-1] * sigmoid_derivative(activations[-1])]

    # Calcular errores y deltas para capas ocultas
    for i in range(num_layers - 2, -1, -1):
        error = deltas[-1].dot(weights[i + 1].T)
        delta = error * sigmoid_derivative(activations[i + 1])
        errors.append(error)
        deltas.append(delta)

    # Invertir deltas para que estén en el orden correcto
    deltas = deltas[::-1]

    # Actualizar pesos y sesgos usando gradientes y deltas
    for i in range(num_layers - 1, -1, -1):
        weights[i] += learning_rate * activations[i].T.dot(deltas[i])
        biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    # Calcular el error total
    total_error = np.mean(np.abs(errors[-1]))
    return weights, biases, total_error

# Entrenamiento del perceptrón multicapa
def train_multilayer_perceptron(X, y, layers, learning_rate, epochs):
    num_samples, num_features = X.shape
    weights, biases = initialize_weights(layers)

    for _ in range(epochs):
        for i in range(num_samples):
            input_data = X[i].reshape(1, -1)
            target_output = y[i].reshape(1, -1)

            activations = feed_forward(input_data, weights, biases)
            weights, biases, _ = backpropagation(input_data, target_output, activations, weights, biases, learning_rate)

    return weights, biases

# Clasificación usando el perceptrón multicapa entrenado
def predict_multilayer_perceptron(X, weights, biases):
    activations = feed_forward(X, weights, biases)
    predictions = (activations[-1] > 0.5).astype(int)
    return predictions

# Cargar datos desde el archivo CSV
def load_data(file_name):
    data = pd.read_csv(file_name, header=None)

    # Reemplazar los valores -1 por 0 en las últimas 3 columnas
    data.iloc[:, -3:] = data.iloc[:, -3:].applymap(lambda x: 0 if x == -1 else x)

    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return X, y

# Dividir los datos en conjuntos de entrenamiento y validación
def split_train_validation(X, y, train_size=0.8):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - train_size, random_state=42)
    return X_train, X_val, y_train, y_val

# Validación cruzada leave-k-out
def leave_k_out_cross_validation(X, y, layers, learning_rate, epochs, k=5):
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    k_fold = num_samples // k

    error_values = []

    for i in range(0, num_samples, k_fold):
        validation_indices = indices[i:i + k_fold]
        train_indices = np.concatenate([indices[:i], indices[i + k_fold:]])

        X_train, X_val = X[train_indices], X[validation_indices]
        y_train, y_val = y[train_indices], y[validation_indices]

        weights, biases = train_multilayer_perceptron(X_train, y_train, layers, learning_rate, epochs)
        predictions = predict_multilayer_perceptron(X_val, weights, biases)

        accuracy = accuracy_score(y_val, predictions)
        error = 1 - accuracy
        error_values.append(error)

    average_error = np.mean(error_values)
    std_deviation = np.std(error_values)

    return average_error, std_deviation

# Función principal
def main():
    input_size = 4  # Cantidad de características de entrada
    output_size = 1  # Clasificación binaria (0 o 1)

    # Define la arquitectura de la red neuronal
    layers = [4, 8, 8, 1]

    # Hiperparámetros ajustados
    hidden_layers = 2  # Número de capas ocultas
    neurons_per_hidden_layer = 8  # Neuronas por capa oculta
    learning_rate = 0.1
    epochs = 1000  # Aumenta el número de épocas

    # Cargar y preparar los datos (reemplaza 'irisbin.csv' con tu archivo de datos)
    X, y = load_data('irisbin.csv')

    # Codifica las etiquetas de -1 y 1 a 0 y 1
    y = (y + 1) / 2

    # Normaliza los datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divide los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = split_train_validation(X, y, train_size=0.8)

    # Entrenar el perceptrón multicapa en el conjunto de entrenamiento
    trained_weights, trained_biases = train_multilayer_perceptron(X_train, y_train, layers, learning_rate, epochs)

    # Realizar predicciones en el conjunto de validación
    predictions = predict_multilayer_perceptron(X_val, trained_weights, trained_biases)

    # Calcular el error de clasificación en el conjunto de validación
    validation_accuracy = accuracy_score(y_val, predictions)
    validation_error = 1 - validation_accuracy

    print(f"Error de clasificación en el conjunto de validación: {validation_error * 100:.2f}%")

    # Realizar validación cruzada leave-k-out
    k = 5  # Puedes ajustar el valor de k según tus necesidades
    avg_error, std_dev = leave_k_out_cross_validation(X, y, layers, learning_rate, epochs, k)

    print(f"Promedio del error de clasificación leave-{k}-out: {avg_error * 100:.2f}%")
    print(f"Desviación estándar del error de clasificación leave-{k}-out: {std_dev * 100:.2f}%")

if __name__ == "__main__":
    main()
