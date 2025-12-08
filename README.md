# Mundial Predictor

Este proyecto predice resultados de partidos de fútbol utilizando un modelo de **RandomForestRegressor** entrenado con datos históricos.

El modelo tiene las siguientes características:
1.  **Predicción de Goles Exactos**: Predice cuántos goles anotará cada equipo.
2.  **Sin sesgo de localía**: Trata los partidos de forma simétrica (Equipo A vs Equipo B).
3.  **Ponderación Temporal**: Da mucho más peso a partidos recientes que a los antiguos.
4.  **Manejo de Eliminatorias**: Si es un partido eliminatorio y termina en empate, simula una definición por penales (aleatorio).

## Estructura

-   `matches.csv`: Dataset histórico.
-   `train_model.py`: Script de entrenamiento. Genera `football_model.joblib` y `team_encoder.joblib`.
-   `main.py`: API en FastAPI.
-   `requirements.txt`: Dependencias.

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

Para generar el modelo:

```bash
python train_model.py
```

## Ejecución de la API

```bash
uvicorn main:app --reload
```

## Uso de la API

Endpoint: `POST /predict`

**Ejemplo de Request (Partido Normal):**
```json
{
  "team1": "Brazil",
  "team2": "Argentina"
}
```

**Ejemplo de Request (Eliminatoria):**
```json
{
  "team1": "Croatia",
  "team2": "Japan",
  "is_knockout": true
}
```

**Ejemplo de Respuesta:**
```json
{
  "match": "Croatia vs Japan",
  "predicted_score": "1 - 1",
  "is_knockout": true,
  "details": {
    "Croatia_goals_raw": 1.1,
    "Japan_goals_raw": 0.9,
    "winner": "Empate (Croatia gana en penales)"
  },
  "qualified_team": "Croatia"
}
```

