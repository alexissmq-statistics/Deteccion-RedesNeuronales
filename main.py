from ultralytics import YOLO
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# Ruta al archivo de configuración del dataset
data_yaml = os.path.join(BASE_DIR, "dataset.yaml")

# Inicializar el modelo YOLOv8n
model = YOLO('yolov8n.pt')  # Cargar modelo pre-entrenado

# Configurar parámetros de entrenamiento adaptados para hardware limitado
results = model.train(
    data=data_yaml,
    epochs=20,          # Puedes ajustar según necesidad
    imgsz=416,           # Tamaño reducido para menor consumo de memoria
    batch=8,             # Batch pequeño para menor consumo de VRAM
    workers=1,           # Menos workers para evitar sobrecarga
    patience=20,         # Early stopping
    project=results_dir,
    name="yolov8n_torre_perforacion",
    pretrained=True,
    optimizer="AdamW",   # Optimizador eficiente para memoria
    cache=False,         # Deshabilitar caché si tienes poca RAM
    device='cpu',
    amp=True,            # Habilitar precisión mixta para ahorro de memoria
    close_mosaic=10,     # Desactivar mosaico en últimas epochs para estabilidad
    max_det=100,         # Limitar detecciones máximas
    rect=False,          # No usar imágenes rectangulares para batch
    plots=True           # Generar gráficos de entrenamiento
)

# Validar el modelo entrenado
model.val()