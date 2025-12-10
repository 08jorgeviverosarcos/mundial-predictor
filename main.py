import os
import random
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Mundial Predictor",
    description="API para predecir resultados exactos de fútbol (XGBoost + Gemini)"
)

# Configuración de CORS
origins = [
    "https://mundial2026.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nuevas rutas de artefactos
MODEL_PATH = "fifa_2026_model.joblib"
META_PATH = "fifa_2026_meta.joblib"

# Configuración de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model = None
meta_data = None
last_elos = {}

class MatchRequest(BaseModel):
    team1: str
    team2: str
    is_knockout: bool = False

class BatchMatchRequest(BaseModel):
    matches: list[MatchRequest]

@app.on_event("startup")
def load_artifacts():
    global model, meta_data, last_elos
    
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta_data = joblib.load(META_PATH)
        
        # Cargar ELOs en un dict rápido
        # meta_data["last_elos"] es un DataFrame
        elo_df = meta_data["last_elos"]
        last_elos = dict(zip(elo_df["team"], elo_df["elo"]))
        
        print(f"✅ Modelo XGBoost y Metadatos cargados. {len(last_elos)} equipos con ELO.")
    else:
        print(f"⚠️ ADVERTENCIA: No se encontraron {MODEL_PATH} o {META_PATH}. El modelo local fallará.")

def get_team_elo(team_name: str):
    """Obtiene el último ELO conocido o usa un promedio global por defecto (1500)"""
    norm_name = team_name.strip().upper()
    return last_elos.get(norm_name, 1500.0)

def goals_to_int(x: float) -> int:
    """Convierte predicción Poisson (float) a entero."""
    if x < 0.5: return 0
    elif x < 1.5: return 1
    elif x < 2.5: return 2
    elif x < 3.5: return 3
    else: return 4

def _predict_single_match_local(team1_name: str, team2_name: str, is_knockout: bool):
    global model, meta_data
    
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    t1_norm = team1_name.strip().upper()
    t2_norm = team2_name.strip().upper()
    
    # 1. Obtener Features
    elo1 = get_team_elo(t1_norm)
    elo2 = get_team_elo(t2_norm)
    
    # ¿Quién es local?
    # En el modelo, "is_host" era 1 si el equipo jugaba en casa Y no era campo neutral.
    # Para el Mundial 2026, los hosts son USA, Mexico, Canada.
    hosts = [h.upper() for h in meta_data.get("hosts", [])]
    
    is_host1 = 1 if t1_norm in hosts else 0
    is_host2 = 1 if t2_norm in hosts else 0
    
    # Valores por defecto para rolling stats (usamos el promedio global del entrenamiento)
    # En producción idealmente tendrías un servicio de stats en vivo, pero aquí usamos el promedio histórico
    # para no penalizar injustamente a nadie.
    avg_goals_global = meta_data.get("global_avg_goals", 1.3)
    
    # Features esperadas por el modelo:
    # ["elo", "opp_elo", "elo_diff", "is_host", "rolling_goals_scored", "rolling_goals_conceded"]
    
    # Perspectiva Team 1
    features_t1 = pd.DataFrame([{
        "elo": elo1,
        "opp_elo": elo2,
        "elo_diff": elo1 - elo2,
        "is_host": is_host1,
        "rolling_goals_scored": avg_goals_global,   # Asumimos forma promedio
        "rolling_goals_conceded": avg_goals_global  # Asumimos defensa promedio
    }])
    
    # Perspectiva Team 2
    features_t2 = pd.DataFrame([{
        "elo": elo2,
        "opp_elo": elo1,
        "elo_diff": elo2 - elo1,
        "is_host": is_host2,
        "rolling_goals_scored": avg_goals_global,
        "rolling_goals_conceded": avg_goals_global
    }])
    
    # Predicción
    pred_t1 = float(model.predict(features_t1)[0])
    pred_t2 = float(model.predict(features_t2)[0])
    
    score1 = goals_to_int(pred_t1)
    score2 = goals_to_int(pred_t2)
    
    winner_label = "Empate"
    if score1 > score2:
        winner_label = team1_name
    elif score2 > score1:
        winner_label = team2_name
        
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

@app.get("/")
def read_root():
    return {"message": "Mundial Predictor API v2 (XGBoost ELO + Gemini)"}

# === PROMPTS GEMINI (Igual que antes) ===

def format_gemini_prompt(matches: list[MatchRequest]) -> str:
    matches_data_lines = []
    for i, m in enumerate(matches):
        t1_clean = m.team1.strip()
        t2_clean = m.team2.strip()
        matches_data_lines.append(f"{i}|{t1_clean}|{t2_clean}")
    matches_data_str = "\n".join(matches_data_lines)
    
    return f"""FIFA 26 Bulk Simulation.
Format: ID|Home|Away.
Stage: FIFA World Cup 2026.
Task: Predict realistic final scores.
IMPORTANT: Rely ENTIRELY on your internal knowledge of real-world football.
Output Format: ID|HomeScore|AwayScore
STRICT INSTRUCTION: Output ONLY the requested format. NO markdown. One line per match.
Example:
0|2|1
1|1|1
Matches to predict:
{matches_data_str}"""

def parse_gemini_response(text: str, original_matches: list[MatchRequest]):
    results = []
    clean_text = text.replace("```csv", "").replace("```", "").strip()
    lines = clean_text.split("\n")
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
                
    for i, m in enumerate(original_matches):
        if i in predictions_map:
            score1, score2 = predictions_map[i]
            winner_label = "Empate"
            if score1 > score2: winner_label = m.team1
            elif score2 > score1: winner_label = m.team2
            
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
                "details": {"winner": winner_label, "source": "Gemini"}
            }
            if m.is_knockout: item["qualified_team"] = qualified
            results.append(item)
        else:
            # FALLBACK A LOCAL POR ITEM
            try:
                res = _predict_single_match_local(m.team1, m.team2, m.is_knockout)
                item = {
                    "match": f"{m.team1} vs {m.team2}",
                    "team1_score": res["score1"],
                    "team2_score": res["score2"],
                    "details": {"winner": res["winner_label"], "source": "Local Model (Fallback)"}
                }
                if m.is_knockout: item["qualified_team"] = res["qualified"]
                results.append(item)
            except Exception as e:
                results.append({"match": f"{m.team1} vs {m.team2}", "error": str(e)})
                
    return results

# === ENDPOINTS ===

@app.post("/predict")
def predict_score(request: MatchRequest):
    try:
        res = _predict_single_match_local(request.team1, request.team2, request.is_knockout)
        response = {
            "match": f"{request.team1} vs {request.team2}",
            "team1_score": res["score1"],
            "team2_score": res["score2"],
            "is_knockout": request.is_knockout,
            "details": {
                f"{request.team1}_goals_raw": round(res["pred_t1_raw"], 2),
                f"{request.team2}_goals_raw": round(res["pred_t2_raw"], 2),
                "winner": res["winner_label"],
                "source": "Local XGBoost"
            },
        }
        if request.is_knockout:
            response["qualified_team"] = res["qualified"]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-gemini")
def predict_gemini(request: MatchRequest):
    if GEMINI_API_KEY:
        try:
            model_gemini = genai.GenerativeModel("gemini-2.5-flash-lite")
            prompt = format_gemini_prompt([request])
            response = model_gemini.generate_content(prompt)
            parsed = parse_gemini_response(response.text, [request])
            if parsed and "error" not in parsed[0]:
                return parsed[0]
        except Exception as e:
            print(f"Gemini error: {e}")
            pass
            
    # Fallback directo a endpoint local logic
    return predict_score(request)

@app.post("/predict-batch")
def predict_batch(request: BatchMatchRequest):
    results = []
    for m in request.matches:
        try:
            res = _predict_single_match_local(m.team1, m.team2, m.is_knockout)
            item = {
                "match": f"{m.team1} vs {m.team2}",
                "team1_score": res["score1"],
                "team2_score": res["score2"],
                "details": {
                    "winner": res["winner_label"],
                    "source": "Local XGBoost"
                }
            }
            if m.is_knockout: item["qualified_team"] = res["qualified"]
            results.append(item)
        except Exception as e:
            results.append({"match": f"{m.team1} vs {m.team2}", "error": str(e)})
    return {"total_matches": len(results), "results": results}

@app.post("/predict-batch-gemini")
def predict_batch_gemini(request: BatchMatchRequest):
    if GEMINI_API_KEY:
        try:
            model_gemini = genai.GenerativeModel("gemini-2.5-flash-lite")
            prompt = format_gemini_prompt(request.matches)
            response = model_gemini.generate_content(prompt)
            parsed = parse_gemini_response(response.text, request.matches)
            return {"total_matches": len(parsed), "results": parsed}
        except Exception as e:
            print(f"Gemini batch error: {e}")
            pass
            
    # Fallback a batch local
    return predict_batch(request)
