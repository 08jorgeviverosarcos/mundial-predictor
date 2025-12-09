import os
import random
import numpy as np
import pandas as pd
import joblib
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Mundial Predictor",
    description="API para predecir resultados exactos de fútbol"
)

MODEL_PATH = "football_model.joblib"
ENCODER_PATH = "team_encoder.joblib"
MATCHES_PATH = "matches.csv"  # usamos el mismo dataset para calcular stats

# Configuración de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("ADVERTENCIA: GEMINI_API_KEY no encontrada en variables de entorno.")

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


def _predict_single_match_local(team1_name: str, team2_name: str, is_knockout: bool):
    """Lógica interna para predecir con el modelo local"""
    global model, encoder
    
    t1_norm = team1_name.strip().upper()
    t2_norm = team2_name.strip().upper()

    # Validar equipos
    known_teams = set(encoder.classes_)
    if t1_norm not in known_teams:
        raise HTTPException(status_code=400, detail=f"Equipo desconocido: {team1_name}")
    if t2_norm not in known_teams:
        raise HTTPException(status_code=400, detail=f"Equipo desconocido: {team2_name}")

    t1_code = encoder.transform([t1_norm])[0]
    t2_code = encoder.transform([t2_norm])[0]

    # Stats ofensivas e historial (de nuestro mapa)
    t1_avg_goals, t1_matches = get_team_features(t1_norm)
    t2_avg_goals, t2_matches = get_team_features(t2_norm)

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
        winner_label = team1_name
    elif score2 > score1:
        winner_label = team2_name

    # Lógica knockout
    qualified = None
    if is_knockout:
        if winner_label == "Empate":
            qualified = random.choice([team1_name, team2_name])
            winner_label = f"Empate ({qualified} gana en penales)"
        else:
            qualified = winner_label

    return {
        "score1": score1,
        "score2": score2,
        "pred_t1_raw": pred_t1,
        "pred_t2_raw": pred_t2,
        "winner_label": winner_label,
        "qualified": qualified
    }


@app.post("/predict")
def predict_score(request: MatchRequest):
    global model, encoder
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    result = _predict_single_match_local(request.team1, request.team2, request.is_knockout)
    
    response = {
        "match": f"{request.team1} vs {request.team2}",
        "team1_score": result["score1"],
        "team2_score": result["score2"],
        "is_knockout": request.is_knockout,
        "details": {
            f"{request.team1}_goals_raw": round(result["pred_t1_raw"], 2),
            f"{request.team2}_goals_raw": round(result["pred_t2_raw"], 2),
            "winner": result["winner_label"],
        },
    }

    if request.is_knockout:
        response["qualified_team"] = result["qualified"]

    return response


@app.post("/predict-batch")
def predict_batch(request: BatchMatchRequest):
    global model, encoder

    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    results = []
    
    # Podríamos optimizar usando el _predict_single_match_local o hacerlo en batch real
    # Por consistencia y manejo de errores, iteramos:
    
    for match in request.matches:
        try:
            res = _predict_single_match_local(match.team1, match.team2, match.is_knockout)
            item = {
                "match": f"{match.team1} vs {match.team2}",
                "team1_score": res["score1"],
                "team2_score": res["score2"],
                "details": {
                    f"{match.team1}_goals_raw": round(res["pred_t1_raw"], 2),
                    f"{match.team2}_goals_raw": round(res["pred_t2_raw"], 2),
                    "winner": res["winner_label"],
                },
            }
            if match.is_knockout:
                item["qualified_team"] = res["qualified"]
            results.append(item)
        except HTTPException:
            # Si falla un equipo (no existe), lo agregamos como error o saltamos
            # Aquí pondremos valores nulos para indicar fallo
            results.append({
                "match": f"{match.team1} vs {match.team2}",
                "error": "Equipo desconocido"
            })

    return {
        "total_matches": len(results),
        "results": results
    }


# === SERVICIOS CON GEMINI ===

def format_gemini_prompt(matches: list[MatchRequest]) -> str:
    # Preparar datos
    matches_data_lines = []
    is_knockout_global = False # Asumimos que si hay uno knockout, el contexto es knockout o mixto
    
    for i, m in enumerate(matches):
        if m.is_knockout:
            is_knockout_global = True
        
        # Obtener "Rat" (Rating) = team_avg_goals * 10 (aprox, para que se vea como 15, 20, etc)
        # O simplemente usamos el avg goals crudo.
        t1_avg, _ = get_team_features(m.team1)
        t2_avg, _ = get_team_features(m.team2)
        
        # ID|Home|Rat|Away|Rat
        # Usamos i como ID
        line = f"{i}|{m.team1}|{t1_avg:.2f}|{m.team2}|{t2_avg:.2f}"
        matches_data_lines.append(line)
        
    matches_data_str = "\n".join(matches_data_lines)
    
    prompt = f"""FIFA 26 Bulk Sim.

Format: ID|Home|Rat|Away|Rat.

Stage: {'Knockout(No draws)' if is_knockout_global else 'Group(Draws ok)'}.

Task: Predict scores based on Rat.
Output Format: ID|HomeScore|AwayScore

Data:

{matches_data_str}"""
    return prompt

def parse_gemini_response(text: str, original_matches: list[MatchRequest]):
    """
    Parsea la respuesta de texto de Gemini en formato ID|Score1|Score2
    """
    results = []
    lines = text.strip().split("\n")
    
    # Mapa por ID para reordenar si es necesario
    predictions_map = {}
    
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3:
            try:
                mid = int(parts[0].strip())
                s1 = int(parts[1].strip())
                s2 = int(parts[2].strip())
                predictions_map[mid] = (s1, s2)
            except ValueError:
                continue
                
    # Reconstruir respuesta
    for i, m in enumerate(original_matches):
        if i in predictions_map:
            score1, score2 = predictions_map[i]
            
            winner_label = "Empate"
            if score1 > score2:
                winner_label = m.team1
            elif score2 > score1:
                winner_label = m.team2
                
            qualified = None
            if m.is_knockout:
                if winner_label == "Empate":
                    qualified = random.choice([m.team1, m.team2])
                    winner_label = f"Empate ({qualified} gana en penales)"
                else:
                    qualified = winner_label
            
            item = {
                "match": f"{m.team1} vs {m.team2}",
                "team1_score": score1,
                "team2_score": score2,
                "details": {
                    "winner": winner_label,
                    "source": "Gemini"
                }
            }
            if m.is_knockout:
                item["qualified_team"] = qualified
            results.append(item)
        else:
             results.append({
                "match": f"{m.team1} vs {m.team2}",
                "error": "No prediction from Gemini"
            })
            
    return results

@app.post("/predict-gemini")
def predict_gemini(request: MatchRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API Key no configurada.")
        
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        prompt = format_gemini_prompt([request])
        
        response = model_gemini.generate_content(prompt)
        parsed_results = parse_gemini_response(response.text, [request])
        
        if not parsed_results:
             raise HTTPException(status_code=500, detail="Error parseando respuesta de Gemini")
             
        return parsed_results[0]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch-gemini")
def predict_batch_gemini(request: BatchMatchRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API Key no configurada.")
        
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        prompt = format_gemini_prompt(request.matches)
        
        response = model_gemini.generate_content(prompt)
        parsed_results = parse_gemini_response(response.text, request.matches)
        
        return {
            "total_matches": len(parsed_results),
            "results": parsed_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
