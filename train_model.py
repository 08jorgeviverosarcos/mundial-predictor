import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib


def main():
    # 1. Cargar datos
    print("Cargando dataset...")
    df = pd.read_csv("matches.csv")

    # 2. Preprocesamiento básico
    print("Preprocesando datos...")

    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida en matches.csv: {col}")

    # Eliminar partidos sin resultado o sin fecha
    df = df.dropna(subset=["home_score", "away_score", "date"])

    # Normalizar tipos y nombres
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].astype(str).str.strip().str.upper()
    df["away_team"] = df["away_team"].astype(str).str.strip().str.upper()

    # 3. Dataset simétrico (cada partido -> 2 filas)
    print("Construyendo dataset simétrico (team / opponent)...")

    df1 = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals": df["home_score"],
    })

    df2 = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals": df["away_score"],
    })

    full_df = pd.concat([df1, df2], ignore_index=True)

    # Seguridad: quitar filas raras
    full_df = full_df.dropna(subset=["team", "opponent", "goals", "date"])

    # 4. Estadísticas históricas por equipo (fuerza ofensiva)
    print("Calculando estadísticas históricas de equipos (ataque)...")

    team_stats = (
        full_df
        .groupby("team")["goals"]
        .agg(["mean", "count"])
        .reset_index()
    )
    team_stats.columns = ["team", "team_avg_goals", "team_matches"]

    # Merge stats para team
    full_df = full_df.merge(team_stats, on="team", how="left")

    # Stats para opponent
    opp_stats = team_stats.copy()
    opp_stats.columns = ["opponent", "opp_avg_goals", "opp_matches"]
    full_df = full_df.merge(opp_stats, on="opponent", how="left")

    # Medias globales por si falta algo
    global_avg_goals = full_df["goals"].mean()
    global_matches_median = full_df[["team_matches", "opp_matches"]].median().median()

    # Rellenar NA de forma segura (sin inplace encadenado)
    full_df["team_avg_goals"] = full_df["team_avg_goals"].fillna(global_avg_goals)
    full_df["opp_avg_goals"] = full_df["opp_avg_goals"].fillna(global_avg_goals)
    full_df["team_matches"] = full_df["team_matches"].fillna(global_matches_median)
    full_df["opp_matches"] = full_df["opp_matches"].fillna(global_matches_median)

    # Limitar matches para evitar valores ridículos
    full_df["team_matches"] = full_df["team_matches"].clip(0, 300)
    full_df["opp_matches"] = full_df["opp_matches"].clip(0, 300)

    # 5. Codificar equipos
    print("Codificando equipos (LabelEncoder)...")

    le = LabelEncoder()
    all_teams = pd.concat([full_df["team"], full_df["opponent"]]).unique()
    le.fit(all_teams)

    full_df["team_code"] = le.transform(full_df["team"])
    full_df["opponent_code"] = le.transform(full_df["opponent"])

    # 6. Ponderación por recencia
    print("Calculando pesos por recencia...")
    max_date = full_df["date"].max()
    full_df["days_diff"] = (max_date - full_df["date"]).dt.days

    # Decaimiento más suave: la historia cuenta, pero los últimos años pesan más
    alpha = 0.0004  # puedes tunearlo si quieres
    full_df["weight"] = np.exp(-alpha * full_df["days_diff"])

    # Por seguridad, si alguien quedó con NaN en weight, lo rellenamos
    full_df["weight"] = full_df["weight"].fillna(full_df["weight"].mean())

    # 7. Features y target
    print("Preparando features y target...")

    feature_cols = [
        "team_code",
        "opponent_code",
        "team_avg_goals",
        "opp_avg_goals",
        "team_matches",
        "opp_matches",
    ]

    X = full_df[feature_cols].astype(float)
    y = full_df["goals"].astype(float)
    sample_weights = full_df["weight"].astype(float)

    # Debug rápido por si acaso
    print("Shape X:", X.shape)
    print("Algún NaN en X?:", X.isna().any().any())
    print("Algún NaN en y?:", np.isnan(y).any())
    print("Algún NaN en weights?:", np.isnan(sample_weights).any())

    # 8. Entrenar modelo
    print("Entrenando modelo de regresión (RandomForestRegressor)...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y, sample_weight=sample_weights)

    # 9. Guardar modelo + encoder + meta
    print("Guardando modelo y encoder...")

    joblib.dump(model, "football_model.joblib")
    joblib.dump(le, "team_encoder.joblib")

    meta = {
        "feature_cols": feature_cols,
        "train_max_date": max_date,
        "global_avg_goals": float(global_avg_goals),
    }
    joblib.dump(meta, "football_model_meta.joblib")

    print("¡Entrenamiento completado!")
    print("n_features_in_ del modelo:", getattr(model, "n_features_in_", None))

    # 10. Importancia de features (solo para info, puede ser NaN si la suma es 0)
    try:
        print("\nImportancia de las features:")
        fi = model.feature_importances_
        for name, imp in sorted(
            zip(feature_cols, fi),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"- {name}: {imp}")
    except Exception as e:
        print("No se pudo calcular feature_importances_:", e)


if __name__ == "__main__":
    main()
