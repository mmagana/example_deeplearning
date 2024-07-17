# Importar Bibliotecas Necesarias:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el Conjunto de Datos: California Housing Dataset
from sklearn.datasets import fetch_california_housing  # Usar California housing dataset

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["PRICE"] = housing.target  # Ajustar target variable name if needed
# Inspección Inicial del Conjunto de Datos
data.head()
data.describe()
# Visualización de la Relación entre Variables
# Adjust feature names as needed
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data["MedInc"], y=data["PRICE"])
plt.title("Relación entre MedInc y el precio")
plt.xlabel("MedInc")
plt.ylabel("Precio")
plt.show()
# División del Conjunto de Datos
X = data[["MedInc"]]  # Adjust feature name as needed
y = data["PRICE"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Entrenamiento del Modelo
model = LinearRegression()
model.fit(X_train, y_train)
# Predicción y Evaluación del Modelo
y_pred = model.predict(X_test)
# Cálculo de Métricas de Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")
# Visualización de los Resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Datos Reales")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicción del Modelo")
plt.title("Regresión Lineal: Predicción vs Datos Reales")
plt.xlabel("MedInc")  # Adjust feature name as needed
plt.ylabel("Precio")
plt.legend()
plt.show()
