import pandas as pd
import statsmodels.api as sm
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Suprimir todos los warnings
warnings.filterwarnings("ignore")

# DataFrame para vino blanco
df_wine = pd.read_csv('winequality-white.csv')
df_wine.hist()

# DataFrame para automóviles
df_auto = pd.read_csv('AutoInsurSweden.csv')
X_auto = df_auto[['x', 'y']]
y_auto = df_auto[['z']]
X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(X_auto, y_auto, random_state=0)
regression_model_auto = LogisticRegression()
# Entrenar el modelo
regression_model_auto.fit(X_train_auto, y_train_auto)
# Realizar predicciones
pred_auto = regression_model_auto.predict(X_test_auto)
pred_series_auto = pd.Series(pred_auto)
# Graficar el histograma
plt.hist(pred_series_auto, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Auto')
plt.show()

# DataFrame para diabetes
df_diabetes = pd.read_csv('Pima-Indians-Diabetes-Dataset.csv')
df_diabetes = df_diabetes.dropna(subset=['Class'])
X_diabetes = df_diabetes.drop('Class', axis=1)
y_diabetes = df_diabetes['Class']
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes,
                                                                                        random_state=0)
regression_model_diabetes = LogisticRegression()
# Entrenar el modelo
regression_model_diabetes.fit(X_train_diabetes, y_train_diabetes)
# Realizar predicciones
pred_diabetes = regression_model_diabetes.predict(X_test_diabetes)
pred_series_diabetes = pd.Series(pred_diabetes)
# Graficar el histograma
plt.hist(pred_series_diabetes, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Diabetes')
plt.show()

# DataFrame para vino blanco (regresión logística)
df_wine_reg = pd.read_csv('winequality-white.csv')
df_wine_reg = df_wine_reg.dropna(subset=['quality'])
X_wine_reg = df_wine_reg.drop('quality', axis=1)
y_wine_reg = df_wine_reg['quality']
X_train_wine_reg, X_test_wine_reg, y_train_wine_reg, y_test_wine_reg = train_test_split(X_wine_reg, y_wine_reg,
                                                                                        random_state=0)
regression_model_wine_reg = LogisticRegression()
# Entrenar el modelo
regression_model_wine_reg.fit(X_train_wine_reg, y_train_wine_reg)
# Realizar predicciones
pred_wine_reg = regression_model_wine_reg.predict(X_test_wine_reg)
pred_series_wine_reg = pd.Series(pred_wine_reg)
# Graficar el histograma
plt.hist(pred_series_wine_reg, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Vino (Regresión Logística)')
plt.show()

# DataFrame para automóviles (K-Nearest Neighbors)
df_auto_knn = pd.read_csv('AutoInsurSweden.csv')
X_auto_knn = df_auto_knn[['x', 'y']]
y_auto_knn = df_auto_knn[['z']]
X_train_auto_knn, X_test_auto_knn, y_train_auto_knn, y_test_auto_knn = train_test_split(X_auto_knn, y_auto_knn,
                                                                                        random_state=0)
scaler_knn = MinMaxScaler()
X_train_auto_knn = scaler_knn.fit_transform(X_train_auto_knn)
X_test_auto_knn = scaler_knn.transform(X_test_auto_knn)
n_neighbors_knn = 7
knn_auto = KNeighborsClassifier(n_neighbors_knn)
knn_auto.fit(X_train_auto_knn, y_train_auto_knn)
pred_auto_knn = knn_auto.predict(X_test_auto_knn)
pred_series_auto_knn = pd.Series(pred_auto_knn)
# Graficar el histograma
plt.hist(pred_series_auto_knn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Auto (K-Nearest Neighbors)')
plt.show()

# DataFrame para diabetes (K-Nearest Neighbors)
df_diabetes_knn = pd.read_csv('Pima-Indians-Diabetes-Dataset.csv')
df_diabetes_knn = df_diabetes_knn.dropna(subset=['Class'])
X_diabetes_knn = df_diabetes_knn.drop('Class', axis=1)
y_diabetes_knn = df_diabetes_knn['Class']
X_train_diabetes_knn, X_test_diabetes_knn, y_train_diabetes_knn, y_test_diabetes_knn = train_test_split(X_diabetes_knn,
                                                                                                    y_diabetes_knn,
                                                                                                    random_state=0)
scaler_diabetes_knn = MinMaxScaler()
X_train_diabetes_knn = scaler_diabetes_knn.fit_transform(X_train_diabetes_knn)
X_test_diabetes_knn = scaler_diabetes_knn.transform(X_test_diabetes_knn)
n_neighbors_diabetes_knn = 7
knn_diabetes = KNeighborsClassifier(n_neighbors_diabetes_knn)
knn_diabetes.fit(X_train_diabetes_knn, y_train_diabetes_knn)
pred_diabetes_knn = knn_diabetes.predict(X_test_diabetes_knn)
pred_series_diabetes_knn = pd.Series(pred_diabetes_knn)
# Graficar el histograma
plt.hist(pred_series_diabetes_knn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Diabetes (K-Nearest Neighbors)')
plt.show()

# DataFrame para vino blanco (K-Nearest Neighbors)
df_wine_knn = pd.read_csv('winequality-white.csv')
df_wine_knn = df_wine_knn.dropna(subset=['quality'])
X_wine_knn = df_wine_knn.drop('quality', axis=1)
y_wine_knn = df_wine_knn['quality']
X_train_wine_knn, X_test_wine_knn, y_train_wine_knn, y_test_wine_knn = train_test_split(X_wine_knn, y_wine_knn,
                                                                                        random_state=0)
scaler_wine_knn = MinMaxScaler()
X_train_wine_knn = scaler_wine_knn.fit_transform(X_train_wine_knn)
X_test_wine_knn = scaler_wine_knn.transform(X_test_wine_knn)
n_neighbors_wine_knn = 7
knn_wine = KNeighborsClassifier(n_neighbors_wine_knn)
knn_wine.fit(X_train_wine_knn, y_train_wine_knn)
pred_wine_knn = knn_wine.predict(X_test_wine_knn)
pred_series_wine_knn = pd.Series(pred_wine_knn)
# Graficar el histograma
plt.hist(pred_series_wine_knn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Vino (K-Nearest Neighbors)')
plt.show()

# DataFrame para automóviles (SVM)
df_auto_svm = pd.read_csv('AutoInsurSweden.csv')
X_auto_svm = df_auto_svm[['x', 'y']]
y_auto_svm = df_auto_svm[['z']]
X_train_auto_svm, X_test_auto_svm, y_train_auto_svm, y_test_auto_svm = train_test_split(X_auto_svm, y_auto_svm,
                                                                                        test_size=0.2, random_state=42)
scaler_svm = StandardScaler()
X_train_auto_svm = scaler_svm.fit_transform(X_train_auto_svm)
X_test_auto_svm = scaler_svm.transform(X_test_auto_svm)
svm_classifier_auto = svm.SVC(kernel='linear')
svm_classifier_auto.fit(X_train_auto_svm, y_train_auto_svm)
y_pred_auto_svm = svm_classifier_auto.predict(X_test_auto_svm)
pred_series_auto_svm = pd.Series(y_pred_auto_svm)
# Graficar el histograma
plt.hist(pred_series_auto_svm, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Auto (SVM)')
plt.show()

# DataFrame para diabetes (SVM)
df_diabetes_svm = pd.read_csv('Pima-Indians-Diabetes-Dataset.csv')
df_diabetes_svm = df_diabetes_svm.dropna(subset=['Class'])
X_diabetes_svm = df_diabetes_svm.drop('Class', axis=1)
y_diabetes_svm = df_diabetes_svm['Class']
X_train_diabetes_svm, X_test_diabetes_svm, y_train_diabetes_svm, y_test_diabetes_svm = train_test_split(X_diabetes_svm,
                                                                                                    y_diabetes_svm,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)
scaler_svm = StandardScaler()
X_train_diabetes_svm = scaler_svm.fit_transform(X_train_diabetes_svm)
X_test_diabetes_svm = scaler_svm.transform(X_test_diabetes_svm)
svm_classifier_diabetes = svm.SVC(kernel='linear')
svm_classifier_diabetes.fit(X_train_diabetes_svm, y_train_diabetes_svm)
y_pred_diabetes_svm = svm_classifier_diabetes.predict(X_test_diabetes_svm)
pred_series_diabetes_svm = pd.Series(y_pred_diabetes_svm)
# Graficar el histograma
plt.hist(pred_series_diabetes_svm, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Diabetes (SVM)')
plt.show()

# DataFrame para vino blanco (SVM)
df_wine_svm = pd.read_csv('winequality-white.csv')
df_wine_svm = df_wine_svm.dropna(subset=['quality'])
X_wine_svm = df_wine_svm.drop('quality', axis=1)
y_wine_svm = df_wine_svm['quality']
X_train_wine_svm, X_test_wine_svm, y_train_wine_svm, y_test_wine_svm = train_test_split(X_wine_svm, y_wine_svm,
                                                                                        test_size=0.2, random_state=42)
scaler_svm = StandardScaler()
X_train_wine_svm = scaler_svm.fit_transform(X_train_wine_svm)
X_test_wine_svm = scaler_svm.transform(X_test_wine_svm)
svm_classifier_wine = svm.SVC(kernel='linear')
svm_classifier_wine.fit(X_train_wine_svm, y_train_wine_svm)
y_pred_wine_svm = svm_classifier_wine.predict(X_test_wine_svm)
pred_series_wine_svm = pd.Series(y_pred_wine_svm)
# Graficar el histograma
plt.hist(pred_series_wine_svm, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Vino (SVM)')
plt.show()

# DataFrame para automóviles (Naive Bayes)
df_auto_nb = pd.read_csv('AutoInsurSweden.csv')
X_auto_nb = df_auto_nb[['x', 'y']]
y_auto_nb = df_auto_nb[['z']]
X_train_auto_nb, X_test_auto_nb, y_train_auto_nb, y_test_auto_nb = train_test_split(X_auto_nb, y_auto_nb,
                                                                                    test_size=0.2, random_state=42)
scaler_nb = StandardScaler()
X_train_auto_nb = scaler_nb.fit_transform(X_train_auto_nb)
X_test_auto_nb = scaler_nb.transform(X_test_auto_nb)
model_auto_nb = GaussianNB()
model_auto_nb.fit(X_auto_nb, y_auto_nb)
yprob_auto_nb = model_auto_nb.predict_proba(X_test_auto_nb)
y_pred_auto_nb = model_auto_nb.predict(X_test_auto_nb)
pred_series_auto_nb = pd.Series(y_pred_auto_nb)
# Graficar el histograma
plt.hist(pred_series_auto_nb, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Auto (Naive Bayes)')
plt.show()

# DataFrame para diabetes (Naive Bayes)
df_diabetes_nb = pd.read_csv('Pima-Indians-Diabetes-Dataset.csv')
df_diabetes_nb = df_diabetes_nb.dropna(subset=['Class'])
X_diabetes_nb = df_diabetes_nb.drop('Class', axis=1)
y_diabetes_nb = df_diabetes_nb['Class']
X_train_diabetes_nb, X_test_diabetes_nb, y_train_diabetes_nb, y_test_diabetes_nb = train_test_split(X_diabetes_nb,
                                                                                                y_diabetes_nb,
                                                                                                test_size=0.2,
                                                                                                random_state=42)
scaler_nb = StandardScaler()
X_train_diabetes_nb = scaler_nb.fit_transform(X_train_diabetes_nb)
X_test_diabetes_nb = scaler_nb.transform(X_test_diabetes_nb)
model_diabetes_nb = GaussianNB()
model_diabetes_nb.fit(X_diabetes_nb, y_diabetes_nb)
yprob_diabetes_nb = model_diabetes_nb.predict_proba(X_test_diabetes_nb)
y_pred_diabetes_nb = model_diabetes_nb.predict(X_test_diabetes_nb)
pred_series_diabetes_nb = pd.Series(y_pred_diabetes_nb)
# Graficar el histograma
plt.hist(pred_series_diabetes_nb, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Diabetes (Naive Bayes)')
plt.show()

# DataFrame para vino blanco (Naive Bayes)
df_wine_nb = pd.read_csv('winequality-white.csv')
df_wine_nb = df_wine_nb.dropna(subset=['quality'])
X_wine_nb = df_wine_nb.drop('quality', axis=1)
y_wine_nb = df_wine_nb['quality']
X_train_wine_nb, X_test_wine_nb, y_train_wine_nb, y_test_wine_nb = train_test_split(X_wine_nb, y_wine_nb,
                                                                                    test_size=0.2, random_state=42)
scaler_nb = StandardScaler()
X_train_wine_nb = scaler_nb.fit_transform(X_train_wine_nb)
X_test_wine_nb = scaler_nb.transform(X_test_wine_nb)
model_wine_nb = GaussianNB()
model_wine_nb.fit(X_wine_nb, y_wine_nb)
yprob_wine_nb = model_wine_nb.predict_proba(X_test_wine_nb)
y_pred_wine_nb = model_wine_nb.predict(X_test_wine_nb)
pred_series_wine_nb = pd.Series(y_pred_wine_nb)
# Graficar el histograma
plt.hist(pred_series_wine_nb, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Vino (Naive Bayes)')
plt.show()

# DataFrame para automóviles (Red Neuronal)
df_auto_nn = pd.read_csv('AutoInsurSweden.csv')
X_auto_nn = df_auto_nn[['x', 'y']]
y_auto_nn = df_auto_nn[['z']]
X_train_auto_nn, X_test_auto_nn, y_train_auto_nn, y_test_auto_nn = train_test_split(X_auto_nn, y_auto_nn,
                                                                                    test_size=0.2, random_state=42)
scaler_nn = StandardScaler()
X_train_auto_nn = scaler_nn.fit_transform(X_train_auto_nn)
X_test_auto_nn = scaler_nn.transform(X_test_auto_nn)
mlp_auto = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp_auto.fit(X_train_auto_nn, y_train_auto_nn)
y_pred_auto_nn = mlp_auto.predict(X_test_auto_nn)
pred_series_auto_nn = pd.Series(y_pred_auto_nn)
# Graficar el histograma
plt.hist(pred_series_auto_nn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Auto (Red Neuronal)')
plt.show()

# DataFrame para diabetes (Red Neuronal)
df_diabetes_nn = pd.read_csv('Pima-Indians-Diabetes-Dataset.csv')
df_diabetes_nn = df_diabetes_nn.dropna(subset=['Class'])
X_diabetes_nn = df_diabetes_nn.drop('Class', axis=1)
y_diabetes_nn = df_diabetes_nn['Class']
X_train_diabetes_nn, X_test_diabetes_nn, y_train_diabetes_nn, y_test_diabetes_nn = train_test_split(X_diabetes_nn,
                                                                                                y_diabetes_nn,
                                                                                                test_size=0.2,
                                                                                                random_state=42)
scaler_nn = StandardScaler()
X_train_diabetes_nn = scaler_nn.fit_transform(X_train_diabetes_nn)
X_test_diabetes_nn = scaler_nn.transform(X_test_diabetes_nn)
mlp_diabetes = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp_diabetes.fit(X_train_diabetes_nn, y_train_diabetes_nn)
y_pred_diabetes_nn = mlp_diabetes.predict(X_test_diabetes_nn)
pred_series_diabetes_nn = pd.Series(y_pred_diabetes_nn)
# Graficar el histograma
plt.hist(pred_series_diabetes_nn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Diabetes (Red Neuronal)')
plt.show()

# DataFrame para vino blanco (Red Neuronal)
df_wine_nn = pd.read_csv('winequality-white.csv')
df_wine_nn = df_wine_nn.dropna(subset=['quality'])
X_wine_nn = df_wine_nn.drop('quality', axis=1)
y_wine_nn = df_wine_nn['quality']
X_train_wine_nn, X_test_wine_nn, y_train_wine_nn, y_test_wine_nn = train_test_split(X_wine_nn, y_wine_nn,
                                                                                    test_size=0.2, random_state=42)
scaler_nn = StandardScaler()
X_train_wine_nn = scaler_nn.fit_transform(X_train_wine_nn)
X_test_wine_nn = scaler_nn.transform(X_test_wine_nn)
mlp_wine = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp_wine.fit(X_train_wine_nn, y_train_wine_nn)
y_pred_wine_nn = mlp_wine.predict(X_test_wine_nn)
pred_series_wine_nn = pd.Series(y_pred_wine_nn)
# Graficar el histograma
plt.hist(pred_series_wine_nn, bins=10)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Histograma de Predicciones - Vino (Red Neuronal)')
plt.show()
