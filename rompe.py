import cv2
import pygame
import mediapipe as mp
import numpy as np
import random
from scipy.spatial import distance as dist
import os

# --- Inicialización de Mediapipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Inicialización de Pygame y Mixer ---
pygame.init()
pygame.mixer.init() # Para los sonidos

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Juego de Rompe Bloques Mediapipe y Pygame")
font = pygame.font.Font(None, 50)

# --- Carga de Sonidos ---
# (Asegúrate de tener una carpeta 'sonidos' con estos archivos)
try:
    pygame.mixer.music.load('sonidos/background.mp3')
    pygame.mixer.music.set_volume(0.3)
except pygame.error:
    print("Advertencia: No se encontró 'sonidos/background.mp3'")

break_sound = pygame.mixer.Sound('sonidos/break.wav') if os.path.exists('sonidos/break.wav') else None

# --- Estados del Juego ---
game_state = "MENU" # "MENU", "PLAYING", "GAME_OVER"
# --- Variables de Puntuación ---
score = 0
WIN_SCORE = 100

# Cargar y escalar la imagen de fondo
background_image = pygame.image.load("fondo.png")
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

# --- Optimización: Pre-cargar y Pre-escalar imágenes de la paleta ---
ORIGINAL_ENCAVA_WIDTH = 150
WIDE_ENCAVA_WIDTH = ORIGINAL_ENCAVA_WIDTH + 100
encava_width, encava_height = ORIGINAL_ENCAVA_WIDTH, 40
base_encava_image = pygame.image.load("encava.png")
encava_image_normal = pygame.transform.scale(base_encava_image, (ORIGINAL_ENCAVA_WIDTH, encava_height))
encava_image_wide = pygame.transform.scale(base_encava_image, (WIDE_ENCAVA_WIDTH, encava_height))
encava_image = encava_image_normal # Imagen actual a usar
encava_y = HEIGHT - 100
# Variables para suavizar el movimiento de la paleta
smoothed_encava_x = WIDTH // 2
encava_x = WIDTH // 2
SMOOTHING_FACTOR = 0.15 # Ajustado para un movimiento suave con la nueva sensibilidad

# --- Variables de Potenciadores (Power-ups) ---
powerups = []
POWERUP_CHANCE = 0.2 # 20% de probabilidad de que aparezca un potenciador
POWERUP_SPEED = 4
POWERUP_DURATION = 7000 # 7 segundos en milisegundos

# Potenciador: Encava Ancha
widen_paddle_active = False
widen_paddle_end_time = 0

# Potenciador: Arepa Lenta
slow_ball_active = False
slow_ball_end_time = 0

# Diccionario para definir los tipos de potenciadores
POWERUP_TYPES = {
    'widen_paddle': {'color': (0, 191, 255)}, # Azul brillante
    'slow_ball':    {'color': (255, 255, 0)}  # Amarillo
}

arepa_image = pygame.image.load("arepa.png")
arepa_image = pygame.transform.scale(arepa_image, (70, 70))
arepa_x, arepa_y = WIDTH // 2, HEIGHT // 2
ORIGINAL_AREPA_SPEED = 5 # Velocidad reducida para hacerlo más fácil
arepa_speed_x, arepa_speed_y = ORIGINAL_AREPA_SPEED, ORIGINAL_AREPA_SPEED 
arepa_radius = 20

pared_image = pygame.image.load("pared.png")
pared_rows, pared_columns = 3, 8 # Reducido para partidas más cortas
pared_width, pared_height = WIDTH // pared_columns, 50
pared_image = pygame.transform.scale(pared_image, (pared_width, pared_height))

# --- Lógica para asignar valores a los bloques que suman 100 ---
def generate_block_values(total_blocks, total_sum, max_value):
    # Empezar con cada bloque valiendo 1 punto
    values = [1] * total_blocks
    current_sum = total_blocks
    
    # Distribuir los puntos restantes aleatoriamente
    points_to_add = total_sum - current_sum
    for _ in range(points_to_add):
        # Elegir un bloque al azar que aún no haya alcanzado el valor máximo
        available_indices = [i for i, v in enumerate(values) if v < max_value]
        if not available_indices:
            break # No hay más bloques a los que se les pueda sumar puntos
        
        random_index = random.choice(available_indices)
        values[random_index] += 1
    
    random.shuffle(values) # Mezclar los valores para que la distribución sea aleatoria
    return values

pareds = [] # Ahora será una lista de diccionarios
block_values = generate_block_values(pared_rows * pared_columns, WIN_SCORE, 5)
value_index = 0
for row in range(pared_rows):
    for col in range(pared_columns):
        rect = pygame.Rect(col * pared_width, row * pared_height, pared_width, pared_height)
        pareds.append({'rect': rect, 'value': block_values[value_index]})
        value_index += 1

# --- Lógica de Detección de Parpadeo ---
EYE_AR_THRESH = 0.20 # Umbral para considerar un parpadeo
EYE_AR_CONSEC_FRAMES = 2 # Frames consecutivos por debajo del umbral para contar como un parpadeo
BLINK_COUNTER = 0
FRAME_COUNTER = 0
LAST_BLINK_TIME = 0
DOUBLE_BLINK_INTERVAL = 1.0 # 1 segundo para detectar un doble parpadeo

# Índices de los landmarks de los ojos de Face Mesh
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye_landmarks):
    # Calcula las distancias euclidianas entre los puntos verticales del ojo
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Calcula la distancia euclidiana entre los puntos horizontales del ojo
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Calcula el aspect ratio del ojo (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

def run_menu_state(results):
    """Gestiona la lógica y el dibujado de la pantalla de menú."""
    global game_state, FRAME_COUNTER, LAST_BLINK_TIME
    screen.blit(background_image, (0, 0))
    start_text = font.render("Parpadea dos veces para empezar", True, (255, 255, 255))
    text_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(start_text, text_rect)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        left_eye = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in LEFT_EYE_IDXS])
        right_eye = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in RIGHT_EYE_IDXS])
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < EYE_AR_THRESH:
            FRAME_COUNTER += 1
        else:
            if FRAME_COUNTER >= EYE_AR_CONSEC_FRAMES:
                current_time = pygame.time.get_ticks() / 1000.0
                if current_time - LAST_BLINK_TIME < DOUBLE_BLINK_INTERVAL:
                    game_state = "PLAYING"
                    # Iniciar música de fondo al empezar a jugar
                    if pygame.mixer.music.get_busy() == False:
                        pygame.mixer.music.play(-1) # -1 para bucle infinito
                LAST_BLINK_TIME = current_time
            FRAME_COUNTER = 0

def run_playing_state(results):
    """Gestiona la lógica y el dibujado del juego en curso."""
    global encava_x, smoothed_encava_x, encava_width, encava_image
    global arepa_x, arepa_y, arepa_speed_x, arepa_speed_y
    global widen_paddle_active, slow_ball_active, widen_paddle_end_time, slow_ball_end_time
    global score, game_state

    # --- Dibujado del Fondo ---
    # Se usa la imagen de fondo estática en lugar de la cámara
    screen.blit(background_image, (0, 0))

    # --- Lógica de Control Ocular ---
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        left_iris = face_landmarks[468]
        right_iris = face_landmarks[473]
        avg_iris_x = (left_iris.x + right_iris.x) / 2.0
        
        # --- Lógica de Sensibilidad Mejorada (Amplificación de Movimiento) ---
        # Definimos una "zona activa" más pequeña que el ancho total de la cámara.
        # Esto significa que no necesitas mover la cabeza de borde a borde.
        ACTIVE_ZONE_START = 0.3  # 30% desde la izquierda
        ACTIVE_ZONE_END = 0.7    # 70% desde la izquierda
        
        # Normalizamos la posición del iris dentro de la zona activa a un rango de 0.0 a 1.0
        # Primero, limitamos el valor para que no se salga de la zona activa
        clamped_x = max(ACTIVE_ZONE_START, min(avg_iris_x, ACTIVE_ZONE_END))
        
        # Luego, calculamos la posición normalizada
        normalized_pos = (clamped_x - ACTIVE_ZONE_START) / (ACTIVE_ZONE_END - ACTIVE_ZONE_START)
        
        target_x = int(normalized_pos * WIDTH) - encava_width // 2
        smoothed_encava_x = (target_x * SMOOTHING_FACTOR) + (smoothed_encava_x * (1 - SMOOTHING_FACTOR))
        encava_x = int(smoothed_encava_x)

    # --- Actualización de Objetos ---
    # Movimiento de la "arepa"
    arepa_x += arepa_speed_x
    arepa_y += arepa_speed_y

    # Colisiones de la "arepa" con los bordes
    if arepa_x - arepa_radius <= 0 or arepa_x + arepa_radius >= WIDTH:
        arepa_speed_x *= -1
    if arepa_y - arepa_radius <= 0:
        arepa_speed_y *= -1
    if arepa_y + arepa_radius >= HEIGHT:
        # Detener la música al perder
        pygame.mixer.music.stop()
        game_state = "GAME_OVER"

    # Límites de la paleta
    encava_x = max(0, min(encava_x, WIDTH - encava_width))

    # Movimiento de potenciadores
    for powerup in powerups[:]:
        powerup['rect'].y += POWERUP_SPEED
        if powerup['rect'].top > HEIGHT:
            powerups.remove(powerup)

    # --- Gestión de Efectos de Potenciadores ---
    current_time = pygame.time.get_ticks()
    if widen_paddle_active and current_time > widen_paddle_end_time:
        widen_paddle_active = False
        encava_width = ORIGINAL_ENCAVA_WIDTH
        encava_image = encava_image_normal
    if slow_ball_active and current_time > slow_ball_end_time:
        slow_ball_active = False
        arepa_speed_x = np.sign(arepa_speed_x) * ORIGINAL_AREPA_SPEED
        arepa_speed_y = np.sign(arepa_speed_y) * ORIGINAL_AREPA_SPEED

    # --- Lógica de Colisiones ---
    arepa_rect = pygame.Rect(arepa_x - arepa_radius, arepa_y - arepa_radius, arepa_radius * 2, arepa_radius * 2)
    encava_rect = pygame.Rect(encava_x, encava_y, encava_width, encava_height)

    # Colisión arepa-paleta
    if arepa_rect.colliderect(encava_rect) and arepa_speed_y > 0:
        arepa_speed_y *= -1

    # Colisión paleta-potenciador
    for powerup in powerups[:]:
        if encava_rect.colliderect(powerup['rect']):
            powerups.remove(powerup)
            if powerup['type'] == 'widen_paddle':
                widen_paddle_active = True
                encava_width = WIDE_ENCAVA_WIDTH
                encava_image = encava_image_wide
                widen_paddle_end_time = pygame.time.get_ticks() + POWERUP_DURATION
            elif powerup['type'] == 'slow_ball':
                if not slow_ball_active:
                    slow_ball_active = True
                    arepa_speed_x /= 2
                    arepa_speed_y /= 2
                slow_ball_end_time = pygame.time.get_ticks() + POWERUP_DURATION

    # Colisión arepa-bloques
    for block in pareds[:]:
        if arepa_rect.colliderect(block['rect']):
            score += block['value']
            pareds.remove(block)
            arepa_speed_y *= -1
            
            # --- Sonido al Romper Bloque ---
            if break_sound:
                break_sound.play()

            if random.random() < POWERUP_CHANCE:
                powerup_type = random.choice(list(POWERUP_TYPES.keys()))
                powerup_info = POWERUP_TYPES[powerup_type]
                pw_rect = pygame.Rect(block['rect'].centerx - 10, block['rect'].y, 20, 20)
                powerups.append({'rect': pw_rect, 'type': powerup_type, 'color': powerup_info['color']})
            if score >= WIN_SCORE:
                game_state = "GAME_OVER"
                # Detener la música al ganar
                pygame.mixer.music.stop()
            break

    # --- Dibujado ---
    screen.blit(encava_image, (encava_x, encava_y))
    screen.blit(arepa_image, (arepa_x - arepa_radius, arepa_y - arepa_radius))
    for block in pareds:
        screen.blit(pared_image, (block['rect'].x, block['rect'].y))
    for powerup in powerups:
        pygame.draw.rect(screen, powerup['color'], powerup['rect'])
    score_text = font.render(f"Puntaje: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

def run_game_over_state():
    """Gestiona la lógica y el dibujado de la pantalla de fin de juego."""
    screen.blit(background_image, (0, 0))
    if score >= WIN_SCORE:
        end_text = font.render("¡GANASTE!", True, (0, 255, 0))
    else:
        end_text = font.render("¡PERDISTE!", True, (255, 0, 0))
    
    text_rect = end_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    score_final_text = font.render(f"Puntaje Final: {score}", True, (255, 255, 255))
    score_rect = score_final_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
    screen.blit(end_text, text_rect)
    screen.blit(score_final_text, score_rect)
    
    restart_text = font.render("Presiona cualquier tecla para reiniciar", True, (255, 255, 255))
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))
    screen.blit(restart_text, restart_rect)

def reset_game():
    """Reinicia todas las variables del juego para una nueva partida."""
    global game_state, score, arepa_x, arepa_y, arepa_speed_x, arepa_speed_y
    global pareds, powerups, widen_paddle_active, slow_ball_active
    global encava_x, smoothed_encava_x, encava_width, encava_image

    game_state = "MENU"
    score = 0

    arepa_x, arepa_y = WIDTH // 2, HEIGHT // 2
    arepa_speed_x, arepa_speed_y = ORIGINAL_AREPA_SPEED, ORIGINAL_AREPA_SPEED

    encava_width = ORIGINAL_ENCAVA_WIDTH
    encava_image = encava_image_normal
    smoothed_encava_x = WIDTH // 2
    encava_x = WIDTH // 2

    powerups.clear()
    widen_paddle_active = False
    slow_ball_active = False

    pareds.clear()
    block_values = generate_block_values(pared_rows * pared_columns, WIN_SCORE, 5)
    value_index = 0
    for row in range(pared_rows):
        for col in range(pared_columns):
            rect = pygame.Rect(col * pared_width, row * pared_height, pared_width, pared_height)
            pareds.append({'rect': rect, 'value': block_values[value_index]})
            value_index += 1

running = True
clock = pygame.time.Clock()
while running:
    # --- Manejo de Eventos ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            # Si estamos en la pantalla de Game Over, cualquier tecla reinicia el juego
            if game_state == "GAME_OVER":
                reset_game()

    # --- Captura y Procesamiento de Cámara ---
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # --- Lógica y Dibujado por Estado ---
    if game_state == "PLAYING":
        run_playing_state(results)
    elif game_state == "MENU":
        run_menu_state(results)
    elif game_state == "GAME_OVER":
        run_game_over_state()

    pygame.display.flip()

cap.release()  
cv2.destroyAllWindows()   
pygame.quit()
