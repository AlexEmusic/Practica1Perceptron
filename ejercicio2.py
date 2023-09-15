import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir la función de activación (función escalón)
def step_function(x):
    return 1 if x >= 0 else -1

# Entrenar un perceptrón simple
def train_perceptron(X, y, learning_rate, epochs):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(epochs):
        for i in range(num_samples):
            prediction = step_function(np.dot(X[i], weights) + bias)
            if prediction != y[i]:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]

    return weights, bias

# Probar el perceptrón entrenado
def test_perceptron(X, weights, bias):
    predictions = [step_function(np.dot(x, weights) + bias) for x in X]
    return predictions

# Lectura de patrones de entrenamiento desde un archivo CSV
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Ejercicio 1 - Entrenar y probar el perceptrón para el problema XOR
def exercise_1():
    X_train, y_train = load_data('XOR_trn.csv')
    X_test, y_test = load_data('XOR_tst.csv')

    learning_rate = float(input("Ingrese la tasa de aprendizaje: "))
    epochs = int(input("Ingrese el número máximo de épocas de entrenamiento: "))

    weights, bias = train_perceptron(X_train, y_train, learning_rate, epochs)
    predictions = test_perceptron(X_test, weights, bias)

    print("Predicciones en el conjunto de prueba:", predictions)

    # Mostrar gráficamente los patrones y la recta que los separa
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.plot([-1, 1], [(-weights[0] - bias) / weights[1], (weights[0] - bias) / weights[1]], 'r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Separación de patrones XOR')
    plt.show()

# Ejercicio 2 - Generar particiones de entrenamiento
def exercise_2(filename, perturbation):
    X, y = load_data("spheres1d10.csv")  # Usar el archivo pasado como argumento

    num_samples = len(X)
    num_train_samples = int(num_samples * 0.8)
    num_test_samples = num_samples - num_train_samples

    num_partitions = int(input("Ingrese la cantidad de particiones: "))  # Solicitar número de particiones al usuario

    for _ in range(num_partitions):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Aplicar perturbación en los datos
        if perturbation:
            num_perturbed_samples = int(num_train_samples * perturbation)
            perturbed_indices = np.random.choice(num_train_samples, num_perturbed_samples, replace=False)
            y_train[perturbed_indices] = -y_train[perturbed_indices]

        learning_rate = float(input("Ingrese la tasa de aprendizaje: "))
        epochs = int(input("Ingrese el número máximo de épocas de entrenamiento: "))

        weights, bias = train_perceptron(X_train, y_train, learning_rate, epochs)
        predictions = test_perceptron(X_test, weights, bias)

        accuracy = np.mean(predictions == y_test)
        print(f"Exactitud en la partición: {accuracy * 100:.2f}%")

        # Mostrar gráficamente los patrones y la recta que los separa
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.plot([-1, 1], [(-weights[0] - bias) / weights[1], (weights[0] - bias) / weights[1]], 'r')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Separación de patrones')
        plt.show()

#Salir del programa
def exit():
  print("Saliendo...")

# Ejecutar los ejercicios
print("1- Ejerciocio 1")
print("2- Ejerciocio 2")
print("3- Salir")
exercise_choice = input("Elija la opcion a ejecutar: ")

if exercise_choice == '1':
    exercise_1()
elif exercise_choice == '2':
    perturbation = float(input("Ingrese el porcentaje de perturbación (0-1) o 0 para ninguno: "))
    exercise_2('spheres1d10.csv', perturbation)
elif exercise_choice == '3':
    exit()
else:
    print("Selección no válida. Por favor, elija 1 o 2.")
