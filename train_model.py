import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Definición de los países anfitriones del Mundial 2026
HOSTS_2026 = ['United States', 'Mexico', 'Canada']

def train_fifa_model(file_path="matches_with_elo.csv"):
    """
    Carga el dataset con ELO, realiza la ingeniería de características
    y entrena un modelo XGBoost para predecir los goles marcados.
    """
    print("🚀 Iniciando el proceso de entrenamiento del Modelo FIFA 2026 (XGBoost)...")

    # 1. Cargar y Limpiar Datos
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que '{file_path}' exista y contenga las columnas de ELO.")
        return

    # Limpieza esencial
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Asegurar la limpieza de los scores y convertir a entero (como discutimos)
    df = df.dropna(subset=['home_score', 'away_score', 'elo_home', 'elo_away'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    # Normalizar nombres de equipos a MAYÚSCULAS para consistencia
    df['home_team'] = df['home_team'].astype(str).str.strip().str.upper()
    df['away_team'] = df['away_team'].astype(str).str.strip().str.upper()

    print(f"Datos cargados. Total de partidos válidos: {len(df)}")
    
    # 2. Ingeniería de Características (Dataset Simétrico + Evitar Data Leakage)
    print("⚙️ Ingeniería de Features: Creando dataset simétrico...")
    
    # =========================================================================
    # PERSPECTIVA HOME (Equipo A)
    # =========================================================================
    df1 = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals": df["home_score"],        # TARGET: Goles que marcamos (Home)
        "opp_goals": df["away_score"],    # Goles que recibimos (Away)
        "elo": df["elo_home"],            # Nuestro ELO
        "opp_elo": df["elo_away"],        # ELO del rival
        
        # is_host: Si el partido no es neutral, el equipo en 'home' tiene ventaja (1)
        # Esto enseña al modelo el "costo" de la localía histórica.
        "is_host": np.where(df['neutral'] == False, 1, 0) 
    })

    # =========================================================================
    # PERSPECTIVA AWAY (Equipo B)
    # =========================================================================
    df2 = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals": df["away_score"],        # TARGET: Goles que marcamos (Away)
        "opp_goals": df["home_score"],    # Goles que recibimos (Home)
        "elo": df["elo_away"],
        "opp_elo": df["elo_home"],
        
        # is_host: El equipo en 'away' nunca tiene la ventaja de localía histórica (0)
        "is_host": 0 
    })

    full_df = pd.concat([df1, df2], ignore_index=True).sort_values('date')

    # Feature 1: ELO Diferencial (El más importante)
    full_df['elo_diff'] = full_df['elo'] - full_df['opp_elo']
    
    # Feature 2: Promedios Rodantes (Rolling Averages) - La forma reciente
    # Esto asegura que el modelo sepa cómo jugamos en los últimos 10 partidos, sin data leakage.
    def get_rolling_stats(group):
        # Promedio de goles marcados en los últimos 10 partidos (shift() es clave)
        group['rolling_goals_scored'] = group['goals'].shift().rolling(window=10, min_periods=1).mean()
        # Promedio de goles recibidos en los últimos 10 partidos
        group['rolling_goals_conceded'] = group['opp_goals'].shift().rolling(window=10, min_periods=1).mean()
        return group

    full_df = full_df.groupby('team', group_keys=False).apply(get_rolling_stats)
    
    # Rellenar valores iniciales (los primeros 10 partidos de la historia) con el promedio global
    global_avg_goals = full_df['goals'].mean()
    full_df[['rolling_goals_scored', 'rolling_goals_conceded']] = full_df[['rolling_goals_scored', 'rolling_goals_conceded']].fillna(global_avg_goals)
    
    # 3. Ponderación por Recencia (Time Decay)
    print("⏳ Aplicando pesos por recencia...")
    max_date = full_df["date"].max()
    full_df["days_diff"] = (max_date - full_df["date"]).dt.days
    alpha = 0.0003  # Se puede tunear. Un valor más bajo mantiene más peso a lo viejo.
    full_df["weight"] = np.exp(-alpha * full_df["days_diff"])
    
    # 4. Definición de Features y Target
    
    # Las features clave que aprendió el modelo:
    feature_cols = [
        "elo",                 # Mi ELO
        "opp_elo",             # ELO del rival
        "elo_diff",            # La diferencia (¡Vital!)
        "is_host",             # Histórica ventaja de localía (para entrenamiento)
        "rolling_goals_scored",
        "rolling_goals_conceded" # Defensa del rival (indirectamente)
    ]

    X = full_df[feature_cols].astype(float)
    y = full_df["goals"].astype(float)
    sample_weights = full_df["weight"].astype(float)
    
    # 5. Backtesting Riguroso (Validación Temporal)
    print("🛡️ Realizando Backtesting (Entrenar pasado, Probar futuro)...")
    
    # Entrenar con todo hasta antes del Mundial de Qatar 2022
    split_date = '2022-11-01' 
    
    train = full_df[full_df['date'] < split_date]
    test = full_df[full_df['date'] >= split_date]

    X_train = train[feature_cols]
    y_train = train["goals"]
    X_test = test[feature_cols]
    y_test = test["goals"]
    train_weights = train["weight"]
    
    # 6. Entrenamiento del Modelo XGBoost (Regresión de Conteo Poisson)
    print("🧠 Entrenando XGBoost Regressor (Optimizado para Conteo de Goles)...")

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        objective='count:poisson',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,  # Definir Early Stopping aquí
        eval_metric='rmse'         # Métrica a monitorear
    )

    model.fit(
        X_train, y_train, 
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)], 
        verbose=100
    )

    # 7. Evaluación del Backtesting
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    print("\n===================================")
    print(f"✅ Backtesting completado (Datos 2022-Actualidad):")
    print(f"Error Raíz Cuadrada Media (RMSE): {rmse:.4f} (Menor es mejor)")
    print(f"Coeficiente de Determinación (R2): {r2:.4f}")
    print("===================================")
    
    # 8. Guardar Modelo y Metadatos
    
    # Guardamos el modelo entrenado con TODOS los datos (Train + Test) para que sea lo más actual posible
    # NOTA: Usamos best_iteration + 1 porque best_iteration es 0-indexed
    best_n = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 1000
    
    final_model = xgb.XGBRegressor(
        n_estimators=best_n,
        learning_rate=0.01,
        max_depth=5,
        objective='count:poisson',
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X, y, sample_weight=sample_weights) # Re-entrenar con TODO el set

    joblib.dump(final_model, "fifa_2026_model.joblib")
    print("💾 Modelo final guardado como 'fifa_2026_model.joblib'")
    
    # Guardar metadatos cruciales para la predicción
    meta = {
        "feature_cols": feature_cols,
        "hosts": HOSTS_2026,
        "global_avg_goals": float(global_avg_goals),
        "last_elos": full_df.drop_duplicates(subset=['team'], keep='last')[['team', 'elo']]
    }
    joblib.dump(meta, "fifa_2026_meta.joblib")
    print("💾 Metadatos guardados (incluye anfitriones y último ELO).")
    
    print("\n🎉 Entrenamiento Finalizado. Tu modelo está listo para predecir el fixture 2026.")


if __name__ == "__main__":
    train_fifa_model()