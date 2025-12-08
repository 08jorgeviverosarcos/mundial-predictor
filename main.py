from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import random
import pandas as pd

app = FastAPI(
    title="Mundial Predictor",
    description="API para predecir resultados exactos de fútbol"
)

MODEL_PATH = "football_model.joblib"
ENCODER_PATH = "team_encoder.joblib"
MATCHES_PATH = "matches.csv"  # usamos el mismo dataset para calcular stats

model = None
encoder = None
team_stats_map = {}
global_avg_goals = 1.0
global_matches_median = 10.0


class MatchRequest(BaseModel):
    team1: str
    team2: str
    is_knockout: bool = False

class BatchMatchRequest(BaseModel):
    matches: list[MatchRequest]


def load_team_stats():
    """
    Calcula estadísticas básicas por equipo a partir de matches.csv
    para replicar la lógica de entrenamiento:
    - team_avg_goals
    - team_matches
    """
    global team_stats_map, global_avg_goals, global_matches_median

    if not os.path.exists(MATCHES_PATH):
        print(f"ADVERTENCIA: No se encontró {MATCHES_PATH}. "
              f"Se usarán valores por defecto para las estadísticas de equipos.")
        team_stats_map = {}
        global_avg_goals = 1.0
        global_matches_median = 10.0
        return

    print("Cargando matches.csv para calcular estadísticas de equipos...")
    df = pd.read_csv(MATCHES_PATH)

    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida en matches.csv: {col}")

    df = df.dropna(subset=["home_score", "away_score"])

    df["home_team"] = df["home_team"].astype(str).str.strip().str.upper()
    df["away_team"] = df["away_team"].astype(str).str.strip().str.upper()

    # Dataset simétrico
    df1 = pd.DataFrame({
        "team": df["home_team"],
        "goals": df["home_score"],
    })
    df2 = pd.DataFrame({
        "team": df["away_team"],
        "goals": df["away_score"],
    })
    full_df = pd.concat([df1, df2], ignore_index=True)

    # Promedios y cantidad de partidos por equipo
    stats = (
        full_df
        .groupby("team")["goals"]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats.columns = ["team", "team_avg_goals", "team_matches"]

    global_avg_goals = full_df["goals"].mean()
    global_matches_median = stats["team_matches"].median()

    # Guardamos en un dict para acceso rápido en predicción
    team_stats_map = {}
    for _, row in stats.iterrows():
        team_name = row["team"]
        team_stats_map[team_name] = {
            "team_avg_goals": float(row["team_avg_goals"]),
            "team_matches": float(row["team_matches"]),
        }

    print("Estadísticas de equipos cargadas correctamente.")


def get_team_features(team_name: str):
    """
    Devuelve (avg_goals, matches) para un equipo.
    Si no existe en el mapa, usa valores globales.
    """
    team_name = team_name.upper().strip()
    stats = team_stats_map.get(team_name)

    if stats is None:
        return global_avg_goals, global_matches_median

    # Clampeamos los matches para evitar valores extremos
    matches = max(0.0, min(stats["team_matches"], 300.0))
    return stats["team_avg_goals"], matches


def goals_to_int(x: float) -> int:
    """
    Convierte goles esperados (float) a un entero razonable.
    Evita que todo sea empate 1-1.
    """
    if x < 0.4:
        return 0
    elif x < 1.4:
        return 1
    elif x < 2.4:
        return 2
    elif x < 3.4:
        return 3
    else:
        # Muy raro >3.4, pero lo dejamos en 4 por si acaso
        return 4


@app.on_event("startup")
def load_artifacts():
    global model, encoder

    # Cargar modelo y encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Modelo de regresión y encoder cargados correctamente.")
    else:
        print("ADVERTENCIA: No se encontraron los archivos del modelo "
              "(football_model.joblib / team_encoder.joblib).")

    # Cargar estadísticas de equipos desde matches.csv
    load_team_stats()


@app.get("/")
def read_root():
    return {
        "message": "Bienvenido al Mundial Predictor API. Usa /predict para predecir el marcador."
    }


@app.post("/predict")
def predict_score(request: MatchRequest):
    global model, encoder
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    # Normalizar nombres
    t1_norm = request.team1.strip().upper()
    t2_norm = request.team2.strip().upper()

    # Validar equipos
    known_teams = set(encoder.classes_)
    if t1_norm not in known_teams:
        raise HTTPException(status_code=400, detail=f"Equipo desconocido: {request.team1}")
    if t2_norm not in known_teams:
        raise HTTPException(status_code=400, detail=f"Equipo desconocido: {request.team2}")

    # Codificar
    t1_code = encoder.transform([t1_norm])[0]
    t2_code = encoder.transform([t2_norm])[0]

    # Stats ofensivas e historial (de nuestro mapa)
    t1_avg_goals, t1_matches = get_team_features(t1_norm)
    t2_avg_goals, t2_matches = get_team_features(t2_norm)

    # Orden de features debe coincidir con el entrenamiento:
    # ["team_code", "opponent_code",
    #  "team_avg_goals", "opp_avg_goals",
    #  "team_matches", "opp_matches"]

    # Goles esperados Team 1 vs Team 2
    x1 = np.array([[t1_code, t2_code, t1_avg_goals, t2_avg_goals, t1_matches, t2_matches]])
    pred_t1 = float(model.predict(x1)[0])

    # Goles esperados Team 2 vs Team 1
    x2 = np.array([[t2_code, t1_code, t2_avg_goals, t1_avg_goals, t2_matches, t1_matches]])
    pred_t2 = float(model.predict(x2)[0])

    # Convertir a marcador entero
    score1 = goals_to_int(pred_t1)
    score2 = goals_to_int(pred_t2)

    # Determinar ganador
    winner_label = "Empate"
    if score1 > score2:
        winner_label = request.team1
    elif score2 > score1:
        winner_label = request.team2

    # Lógica knockout
    qualified = None
    if request.is_knockout:
        if winner_label == "Empate":
            qualified = random.choice([request.team1, request.team2])
            winner_label = f"Empate ({qualified} gana en penales)"
        else:
            qualified = winner_label

    response = {
    "match": f"{request.team1} vs {request.team2}",
    "team1_score": score1,
    "team2_score": score2,
    "is_knockout": request.is_knockout,
    "details": {
        f"{request.team1}_goals_raw": round(pred_t1, 2),
        f"{request.team2}_goals_raw": round(pred_t2, 2),
        "winner": winner_label,
    },
}

    if request.is_knockout:
        response["qualified_team"] = qualified

    return response


@app.post("/predict-batch")
def predict_batch(request: BatchMatchRequest):
    global model, encoder

    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    results = []

    known_teams = set(encoder.classes_)

    # ⚡ Cache rápido de stats para no buscarlos 100 veces
    stats_cache = {}

    def get_cached_features(team_name):
        if team_name not in stats_cache:
            stats_cache[team_name] = get_team_features(team_name)
        return stats_cache[team_name]

    for match in request.matches:
        t1_norm = match.team1.strip().upper()
        t2_norm = match.team2.strip().upper()

        if t1_norm not in known_teams:
            raise HTTPException(status_code=400, detail=f"Equipo desconocido: {match.team1}")
        if t2_norm not in known_teams:
            raise HTTPException(status_code=400, detail=f"Equipo desconocido: {match.team2}")

        t1_code = encoder.transform([t1_norm])[0]
        t2_code = encoder.transform([t2_norm])[0]

        t1_avg_goals, t1_matches = get_cached_features(t1_norm)
        t2_avg_goals, t2_matches = get_cached_features(t2_norm)

        # Vectores de predicción
        x1 = np.array([[t1_code, t2_code, t1_avg_goals, t2_avg_goals, t1_matches, t2_matches]])
        x2 = np.array([[t2_code, t1_code, t2_avg_goals, t1_avg_goals, t2_matches, t1_matches]])

        pred_t1 = float(model.predict(x1)[0])
        pred_t2 = float(model.predict(x2)[0])

        score1 = goals_to_int(pred_t1)
        score2 = goals_to_int(pred_t2)

        winner_label = "Empate"
        if score1 > score2:
            winner_label = match.team1
        elif score2 > score1:
            winner_label = match.team2

        qualified = None
        if match.is_knockout:
            if winner_label == "Empate":
                qualified = random.choice([match.team1, match.team2])
                winner_label = f"Empate ({qualified} gana en penales)"
            else:
                qualified = winner_label

        item = {
            "match": f"{match.team1} vs {match.team2}",
            "team1_score": score1,
            "team2_score": score2,
            "details": {
                f"{match.team1}_goals_raw": round(pred_t1, 2),
                f"{match.team2}_goals_raw": round(pred_t2, 2),
                "winner": winner_label,
            },
        }

        if match.is_knockout:
            item["qualified_team"] = qualified

        results.append(item)

    return {
        "total_matches": len(results),
        "results": results
    }

    
