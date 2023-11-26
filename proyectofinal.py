
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Zoo
url_zoo = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
column_names = ['Animal', 'Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator', 'Toothed', 'Backbone',
                'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic', 'Catsize', 'Class']
df_zoo = pd.read_csv(url_zoo, names=column_names)

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X_zoo = df_zoo.drop(['Animal', 'Class'], axis=1)
y_zoo = df_zoo['Class']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train_zoo, X_test_zoo, y_train_zoo, y_test_zoo = train_test_split(X_zoo, y_zoo, test_size=0.2, random_state=42)

# Definir clasificadores
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machines': SVC(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
}

results = {}

# Iterar sobre clasificadores
for clf_name, clf in classifiers.items():
    # Escalar datos si es necesario
    if clf_name != 'Neural Network':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_zoo)
        X_test_scaled = scaler.transform(X_test_zoo)
    else:
        X_train_scaled, X_test_scaled = X_train_zoo, X_test_zoo

    # Entrenar el clasificador
    clf.fit(X_train_scaled, y_train_zoo)

    # Realizar predicciones
    y_pred = clf.predict(X_test_scaled)

    # Evaluar el rendimiento del clasificador
    accuracy = accuracy_score(y_test_zoo, y_pred)
    precision = precision_score(y_test_zoo, y_pred, average='weighted')
    recall = recall_score(y_test_zoo, y_pred, average='weighted')
    f1 = f1_score(y_test_zoo, y_pred, average='weighted')
    confusion = confusion_matrix(y_test_zoo, y_pred)

    # Almacenar resultados
    results[clf_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion
    }

# Crear DataFrame para visualización
df_results = pd.DataFrame(results).T

# Visualización de barras para Accuracy, Precision, Recall, y F1 Score
df_results[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', rot=45, figsize=(10, 6))
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Classifier')
plt.legend(loc='lower right')
plt.show()

# Visualización de matrices de confusión
for clf_name, metrics in results.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {clf_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()