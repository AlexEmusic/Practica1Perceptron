import numpy as np
import matplotlib.pyplot as plt

def perceptron(input_size):
    # Inicialización de pesos y bias
    weights = np.random.rand(input_size)
    bias = np.random.rand()
    
    return weights, bias

def activation_function(value):
    return 1 if value >= 0 else 0

def train_perceptron(data, labels, learning_rate, max_epochs):
    input_size = data.shape[1]
    weights, bias = perceptron(input_size)

    for epoch in range(max_epochs):
        error = 0
        for i in range(data.shape[0]):
            # Calcular el valor de la neurona
            net_input = np.dot(data[i], weights) + bias
            output = activation_function(net_input)

            # Calcular el error
            delta = labels[i] - output

            # Actualizar pesos y bias
            weights += learning_rate * delta * data[i]
            bias += learning_rate * delta

            error += delta**2

        # Criterio de finalización: Error cuadrático medio
        mse = error / data.shape[0]
        if mse == 0:
            print(f"Entrenamiento finalizado en la época {epoch + 1}")
            break

    return weights, bias

def test_perceptron(weights, bias, test_data):
    predictions = []
    for i in range(test_data.shape[0]):
        net_input = np.dot(test_data[i], weights) + bias
        output = activation_function(net_input)
        predictions.append(output)

    return predictions

def plot_data_and_line(X, y, weights, bias):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Clase 0', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Clase 1', marker='x')
    
    # Recta separadora
    x_values = np.linspace(-2, 2, 100)
    y_values = (-weights[0] * x_values - bias) / weights[1]
    plt.plot(x_values, y_values, label='Recta Separadora', color='red')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Patrones y Recta Separadora')
    plt.show()

# Lectura de patrones de entrenamiento desde archivos CSV
train_data = np.loadtxt('XOR_trn.csv', delimiter=',')
test_data = np.loadtxt('XOR_tst.csv', delimiter=',')

X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Parámetros de entrenamiento
learning_rate = 0.1
max_epochs = 1000

# Entrenamiento del perceptrón
trained_weights, trained_bias = train_perceptron(X_train, y_train, learning_rate, max_epochs)

# Prueba del perceptrón entrenado
predictions = test_perceptron(trained_weights, trained_bias, X_test)

# Mostrar resultados
for i, prediction in enumerate(predictions):
    print(f"Entradas: {X_test[i]}, Salida del perceptrón: {prediction}")

# Visualización gráfica de los patrones y la recta separadora
plot_data_and_line(X_train, y_train, trained_weights, trained_bias)
