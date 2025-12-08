import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

# 1. Cargar datos
print("Cargando dataset...")
df = pd.read_csv('matches.csv')

# 2. Preprocesamiento
print("Preprocesando datos...")
df['date'] = pd.to_datetime(df['date'])

# Definir el objetivo (Target)
# 0: Empate, 1: Gana Local (Home), 2: Gana Visitante (Away)
def get_match_outcome(row):
    if row['home_score'] > row['away_score']:
        return 1 # Gana Home
    elif row['away_score'] > row['home_score']:
        return 2 # Gana Away
    else:
        return 0 # Empate

df['outcome'] = df.apply(get_match_outcome, axis=1)

# Codificar equipos
# Unificamos todos los equipos para tener un solo encoder
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
le = LabelEncoder()
le.fit(all_teams)

# Transformamos las columnas de equipos
df['home_team_code'] = le.transform(df['home_team'])
df['away_team_code'] = le.transform(df['away_team'])

# 3. Ponderación por recencia
# Calculamos la diferencia en días desde el partido hasta hoy (o la fecha más reciente del dataset)
max_date = df['date'].max()
df['days_diff'] = (max_date - df['date']).dt.days

# Aplicamos una función de decaimiento para el peso
# weight = exp(-alpha * days_diff)
# alpha determina qué tan rápido decae la importancia. 
# Un alpha de 0.001 significa que un partido de hace 365 días tiene un peso de exp(-0.365) ~= 0.69
alpha = 0.002
df['weight'] = np.exp(-alpha * df['days_diff'])

print(f"Pesos calculados. Ejemplo (Reciente vs Antiguo):")
print(df[['date', 'days_diff', 'weight']].sort_values('date', ascending=False).head(3))
print(df[['date', 'days_diff', 'weight']].sort_values('date', ascending=True).head(3))

# 4. Entrenamiento
features = ['home_team_code', 'away_team_code']
X = df[features]
y = df['outcome']
weights = df['weight']

print("Entrenando modelo...")
# Usamos RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y, sample_weight=weights)

# 5. Guardar modelo y encoder
print("Guardando modelo y artefactos...")
joblib.dump(model, 'football_model.joblib')
joblib.dump(le, 'team_encoder.joblib')

print("¡Entrenamiento completado exitosamente!")

