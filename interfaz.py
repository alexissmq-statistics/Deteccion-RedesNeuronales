from ultralytics import YOLO
import cv2
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
trained_model = YOLO(os.path.join(BASE_DIR, "results", "yolov8n_torre_perforacion", "weights", "best.pt"))

# Ruta a imágenes o video de prueba
ruta_prueba = os.path.join(BASE_DIR, "dataset", "images", "train", "image (18).jpeg")
# ruta_prueba = os.path.join(BASE_DIR, "dataset", "images", "train", "images (36).jpeg")
# ruta_prueba = os.path.join(BASE_DIR, "dataset", "images", "train", "544720_342273189175985_1448632650_n.jpg")
# ruta_prueba = os.path.join(BASE_DIR, "dataset", "images", "train", "corona (20).jpeg")

# Ejecutar inferencia con configuración
results = trained_model.predict(
    source=ruta_prueba,
    conf=0.45,             # Umbral de confianza
    iou=0.45,              # Umbral IoU para NMS
    max_det=100,           # Detecciones máximas
    line_width=2,          # Grosor de línea para visualización
    show=False,             # Mostrar visualización en tiempo real
    #save=False,             # Guardar imagen con resultados
    #save_conf=True,        # Incluir puntuaciones de confianza en etiquetas
    #save_crop=True,        # Guardar recortes de objetos detectados
    verbose=True           # Mostrar información detallada
)

# Obtener la primera predicción
result = results[0]

# Cargar la imagen original
imagen = cv2.imread(ruta_prueba)

# Dibujar las detecciones en la imagen
for box in result.boxes:
    # Obtener coordenadas (convertir a enteros para dibujar)
    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
    
    # Obtener la clase y confianza
    cls = int(box.cls[0].item())
    conf = box.conf[0].item()
    
    # Obtener el nombre de la clase (si está disponible)
    class_name = result.names[cls] if hasattr(result, 'names') else f"Clase {cls}"
    
    # Dibujar rectángulo y etiqueta
    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(imagen, f"{class_name}: {conf:.2f}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con detecciones
cv2.imshow("Detecciones YOLOv8", imagen)
cv2.waitKey(0)  # Esperar hasta que se presione una tecla
cv2.destroyAllWindows()
