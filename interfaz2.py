from ultralytics import YOLO
import cv2
import torch
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
trained_model = YOLO(os.path.join(BASE_DIR, "results", "yolov8n_torre_perforacion", "weights", "best.pt"))

# Ruta al video de prueba
ruta_video = os.path.join(BASE_DIR, "video1.mp4")

# Abrir el video
cap = cv2.VideoCapture(ruta_video)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {ruta_video}")
    exit()

# Obtener información del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video cargado: {width}x{height} a {fps} FPS, {total_frames} frames totales")

# Variables para calcular FPS de procesamiento
frame_count = 0
start_time = time.time()
fps_start_time = start_time
fps_frame_count = 0
display_fps = 0

# Procesar el video frame por frame
while cap.isOpened():
    # Leer un frame
    ret, frame = cap.read()
    
    # Si no hay más frames, salir del bucle
    if not ret:
        print("Fin del video o error al leer frame.")
        break
    
    frame_count += 1
    fps_frame_count += 1
    
    # Actualizar cálculo de FPS cada segundo
    current_time = time.time()
    if current_time - fps_start_time >= 1.0:
        display_fps = fps_frame_count / (current_time - fps_start_time)
        fps_frame_count = 0
        fps_start_time = current_time
    
    # Mostrar progreso cada 30 frames
    if frame_count % 30 == 0:
        elapsed = current_time - start_time
        percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"Procesando: {frame_count}/{total_frames} frames ({percent_complete:.1f}%), {display_fps:.1f} FPS")
    
    # Realizar inferencia en el frame actual
    results = trained_model.predict(
        source=frame,
        conf=0.3,             # Umbral de confianza
        iou=0.45,             # Umbral IoU para NMS
        max_det=100,          # Detecciones máximas
        line_width=2,         # Grosor de línea para visualización
        verbose=False         # Desactivar información detallada para mejorar rendimiento
    )
    
    # Obtener el resultado del frame actual
    result = results[0]
    
    # Obtener el frame con las detecciones dibujadas
    annotated_frame = result.plot()
    
    # Añadir información de FPS al frame
    cv2.putText(
        annotated_frame, 
        f"FPS: {display_fps:.1f}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Mostrar el frame con detecciones
    cv2.imshow("Detecciones YOLOv8", annotated_frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Procesamiento interrumpido por el usuario.")
        break

# Calcular estadísticas finales
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"Procesamiento completado: {frame_count} frames en {total_time:.2f} segundos ({avg_fps:.2f} FPS promedio)")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
