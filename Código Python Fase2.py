import pandas as pd

ruta = r"C:\Users\Vanessa\OneDrive\Documentos\Dataset.csv"
df = pd.read_csv(ruta)

if len(df.columns) == 1:
    df = pd.read_csv(ruta, sep=';', header=0)

df.columns = ['Date', 'Person_ID', 'Age', 'Gender', 'Total_Sleep_Hours', 'Sleep_Quality', 'Exercise_Mins_Day', 'Caffeine_Intake', 'Screen_Time_Before_Bed', 'Work_Hours', 'Productivity_Score', 'Mood_Score', 'Stress_Level']

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

df = df.drop_duplicates()

df = df.dropna()

df = df[(df['Age'] >= 18) & (df['Age'] <= 60)]

df['Gender'] = df['Gender'].astype('category')

df = df.reset_index(drop=True)

df.to_csv(r"C:\Users\Vanessa\OneDrive\Documentos\Dataset_Limpio.csv", index=False)

print("Limpieza completada")
