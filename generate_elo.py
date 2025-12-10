import pandas as pd
import numpy as np

def expected_result(loc, vis, is_neutral):
    """
    Calcula la probabilidad de victoria del local.
    Si NO es neutral, añadimos 100 puntos de ventaja de campo al local temporalmente.
    """
    home_advantage = 100 if not is_neutral else 0
    dr = (vis - (loc + home_advantage))
    return 1 / (1 + 10 ** (dr / 400))

def update_elo(winner_elo, loser_elo, k_factor=30):
    # Resultado real: 1 para ganador, 0 para perdedor (simplificado)
    # En empate se maneja diferente (0.5), ver lógica abajo
    pass 

def generate_elo_csv():
    print("Cargando matches.csv...")
    df = pd.read_csv("matches.csv")
    
    # 1. Limpieza básica y orden cronológico (VITAL)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Asegurar que neutral sea booleano
    # A veces viene como string "TRUE"/"FALSE", esto lo arregla:
    if df['neutral'].dtype == 'object':
        df['neutral'] = df['neutral'].astype(str).str.upper() == 'TRUE'

    # 2. Inicializar Diccionario de Elo
    elo_dict = {}
    current_elos_home = []
    current_elos_away = []
    
    # Configuración
    STARTING_ELO = 1500
    K_FACTOR = 30  # Cuánto cambia el Elo por partido (puedes bajarlo a 20 para menos volatilidad)
    
    print("Calculando ELO histórico paso a paso...")
    
    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        h_score = row['home_score']
        a_score = row['away_score']
        is_neutral = row['neutral']
        
        # Obtener Elo actual (o inicializar)
        elo_h = elo_dict.get(home, STARTING_ELO)
        elo_a = elo_dict.get(away, STARTING_ELO)
        
        # Guardamos el Elo que tenían ANTES del partido (esto es lo que usará el modelo)
        current_elos_home.append(elo_h)
        current_elos_away.append(elo_a)
        
        # --- CÁLCULO DE ACTUALIZACIÓN ---
        
        # 1. Probabilidad esperada (W_e)
        # Si NO es neutral, el local "juega" como si tuviera +100 elo
        we_h = expected_result(elo_h, elo_a, is_neutral)
        we_a = 1 - we_h
        
        # 2. Resultado Real (W)
        if h_score > a_score:
            w_h, w_a = 1.0, 0.0
        elif h_score < a_score:
            w_h, w_a = 0.0, 1.0
        else:
            w_h, w_a = 0.5, 0.5
            
        # 3. Nuevo Elo
        # Elo Nuevo = Elo Viejo + K * (Resultado Real - Esperado)
        new_elo_h = elo_h + K_FACTOR * (w_h - we_h)
        new_elo_a = elo_a + K_FACTOR * (w_a - we_a)
        
        # Actualizar diccionario
        elo_dict[home] = new_elo_h
        elo_dict[away] = new_elo_a

    # Agregar columnas al DF
    df['elo_home'] = current_elos_home
    df['elo_away'] = current_elos_away
    
    # Guardar
    print("Guardando matches_with_elo.csv...")
    df.to_csv("matches_with_elo.csv", index=False)
    print(f"¡Listo! Último Elo de Argentina: {elo_dict.get('Argentina', 'N/A'):.2f}")
    print(f"Último Elo de Brasil: {elo_dict.get('Brazil', 'N/A'):.2f}")

if __name__ == "__main__":
    generate_elo_csv()