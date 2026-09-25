from ultralytics import YOLO
import cv2
import torch
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
trained_model = YOLO(os.path.join(BASE_DIR, "results", "yolov8n_torre_perforacion", "weights", "best.pt"))

# Configurar la captura de video desde la webcam
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables para calcular FPS
prev_time = 0
curr_time = 0

while True:
    # Leer un frame de la cámara
    success, frame = cap.read()
    
    if not success:
        print("Error: No se pudo leer el frame.")
        break
        
    # Realizar la detección en el frame actual
    results = trained_model.predict(
        source=frame,
        conf=0.4,          # Umbral de confianza
        iou=0.45,          # Umbral IoU para NMS
        max_det=100,       # Detecciones máximas
        line_width=2,      # Grosor de línea para visualización
        verbose=False      # Desactivar información detallada para mayor rendimiento
    )
    
    # Obtener el resultado
    result = results[0]
    
    # Dibujar las detecciones directamente en el frame
    annotated_frame = result.plot()
    
    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    # Mostrar el frame con las detecciones
    cv2.imshow("YOLOv8 Detección en Tiempo Real", annotated_frame)
    
    # Salir si se presiona 'esc'
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
