import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ruta = r"C:\Users\Vanessa\OneDrive\Documentos\Dataset_Limpio.csv"
df = pd.read_csv(ruta)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Total_Sleep_Hours', data=df, estimator='mean', palette='viridis')
plt.title('Promedio de Horas de Sueño por Género')
plt.xlabel('Género')
plt.ylabel('Horas de sueño ')

plt.figure(figsize=(8, 6))
sns.histplot(df['Sleep_Quality'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de la Calidad del Sueño')
plt.xlabel('Calidad del Sueño')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor de Correlaciones')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Stress_Level'], color='orange')
plt.title('Detección de Anomalías en el Nivel de Estrés')
plt.xlabel('Nivel de Estrés')
plt.show()

correlacion = df[['Total_Sleep_Hours', 'Productivity_Score', 'Stress_Level']].corr(numeric_only=True)
print("Correlación entre Horas de Sueño, Productividad y Nivel de Estrés:")
print(correlacion)
