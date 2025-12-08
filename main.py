from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Mundial Predictor", description="API para predecir resultados de fútbol")

# Rutas de archivos
MODEL_PATH = 'football_model.joblib'
ENCODER_PATH = 'team_encoder.joblib'

# Variables globales
model = None
encoder = None

@app.on_event("startup")
def load_artifacts():
    global model, encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Modelo y encoder cargados correctamente.")
    else:
        print("ADVERTENCIA: No se encontraron los archivos del modelo. Ejecuta train_model.py primero.")

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido al Mundial Predictor API. Usa /predict para predecir resultados."}

@app.post("/predict")
def predict_match(request: MatchRequest):
    global model, encoder
    if not model or not encoder:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Contacta al administrador.")
    
    try:
        # Validar equipos
        known_teams = set(encoder.classes_)
        if request.home_team not in known_teams:
            raise HTTPException(status_code=400, detail=f"Equipo desconocido: {request.home_team}. Asegúrate de usar nombres exactos.")
        if request.away_team not in known_teams:
            raise HTTPException(status_code=400, detail=f"Equipo desconocido: {request.away_team}. Asegúrate de usar nombres exactos.")
            
        # Codificar
        home_code = encoder.transform([request.home_team])[0]
        away_code = encoder.transform([request.away_team])[0]
        
        # Predecir
        # Formato de entrada del modelo: [home_team_code, away_team_code]
        features = [[home_code, away_code]]
        
        # Predicción de clase
        prediction = model.predict(features)[0]
        # Probabilidades
        probs = model.predict_proba(features)[0]
        
        # Interpretación (0: Empate, 1: Home, 2: Away)
        result_map = {0: "Empate", 1: request.home_team, 2: request.away_team}
        winner = result_map.get(prediction, "Desconocido")
        
        return {
            "match": f"{request.home_team} vs {request.away_team}",
            "predicted_winner": winner,
            "probabilities": {
                "draw": round(float(probs[0]), 3),
                f"{request.home_team}_win": round(float(probs[1]), 3),
                f"{request.away_team}_win": round(float(probs[2]), 3)
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

