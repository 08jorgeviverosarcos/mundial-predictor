# Mundial Predictor

Este proyecto predice resultados de partidos de fútbol utilizando un modelo de **RandomForestRegressor** entrenado con datos históricos, y también ofrece integración con **Google Gemini** para predicciones generativas.

## Características

1.  **Modelo Local (Random Forest)**:
    *   Predicción de Goles Exactos.
    *   Sin sesgo de localía (Simétrico).
    *   Ponderación Temporal (Mayor peso a partidos recientes).
2.  **Modelo Generativo (Gemini)**:
    *   Usa el LLM de Google para simular resultados basados en estadísticas ("Ratings").
    *   Soporta predicción por lotes.
3.  **Manejo de Eliminatorias**: Simulación de penales en caso de empate en fases eliminatorias.

## Estructura

-   `matches.csv`: Dataset histórico.
-   `train_model.py`: Script de entrenamiento (Local).
-   `main.py`: API en FastAPI.
-   `requirements.txt`: Dependencias.

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Para usar los endpoints de Gemini, necesitas una API Key.
Crea un archivo `.env` basado en `env_example.txt`:

```bash
GEMINI_API_KEY=tu_api_key_real
```

## Entrenamiento (Modelo Local)

```bash
python train_model.py
```

## Ejecución de la API

```bash
uvicorn main:app --reload
```

## Endpoints

### 1. Predicción Local

*   **POST** `/predict`
*   **POST** `/predict-batch`

### 2. Predicción con Gemini

*   **POST** `/predict-gemini`
*   **POST** `/predict-batch-gemini`

**Ejemplo de Request (Batch):**
```json
{
  "matches": [
    { "team1": "Brazil", "team2": "Argentina" },
    { "team1": "France", "team2": "Germany", "is_knockout": true }
  ]
}
```
