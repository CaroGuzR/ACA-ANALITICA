import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

ruta = r"C:\Users\Vanessa\OneDrive\Documentos\Dataset_Limpio.csv"
df = pd.read_csv(ruta)

X = df[['Total_Sleep_Hours', 'Stress_Level', 'Exercise_Mins_Day', 'Caffeine_Intake', 'Screen_Time_Before_Bed']]
y = df['Sleep_Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Resultados del modelo:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)

modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nResultados con Random Forest:")
print(f"MAE: {mae_rf:.2f}")
print(f"MSE: {mse_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.2f}")

modelos = ['Regresión Lineal', 'Random Forest']
mae = [2.50, 2.54]
rmse = [2.87, 2.95]

plt.figure(figsize=(8, 5))
plt.bar(modelos, mae, color=['#4CAF50', '#FF5733'], label='MAE')
plt.bar(modelos, rmse, bottom=mae, color=['#8BC34A', '#FF8C66'], label='RMSE')

plt.title('Comparación de Modelos')
plt.ylabel('Error')
plt.legend()
plt.show()

