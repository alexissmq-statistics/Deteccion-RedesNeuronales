# Detección de Partes de una Torre Petrolera con YOLOv8

Proyecto de detección de objetos que identifica partes de una torre de perforación petrolera a partir de imágenes, video y transmisión de webcam, usando un modelo YOLOv8n (Ultralytics) entrenado sobre un dataset propio.

## Estructura del proyecto

```
.
├── dataset/                  # Imágenes de entrenamiento/validación y sus anotaciones
├── dataset.yaml              # Configuración del dataset para YOLO (rutas y clases)
├── results/                  # Salida del entrenamiento (pesos, métricas, gráficos)
│   └── yolov8n_torre_perforacion/
│       └── weights/
│           └── best.pt       # Mejor checkpoint del modelo entrenado
├── main.py                   # Script de entrenamiento del modelo
├── interfaz.py                # Inferencia sobre una imagen de prueba
├── interfaz2.py                # Inferencia sobre un archivo de video
├── InterfazVideoYolo.py       # Inferencia en tiempo real desde webcam
├── video1.mp4                 # Video de prueba (opcional)
├── yolov8n.pt                 # Pesos base preentrenados de YOLOv8 nano
└── requirements.txt           # Dependencias del proyecto
```

## Requisitos

- Python 3.9+
- Las librerías listadas en `requirements.txt` (principalmente `ultralytics`, `opencv-python`, `torch`)

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-de-tu-repo>
cd <nombre-del-repo>

# 2. Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate      # En Windows
source venv/bin/activate   # En Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** el entorno virtual (`venv/`) no se incluye en el repositorio. Cada persona que clone el proyecto debe crearlo y luego instalar las dependencias con el comando anterior.

## Dataset

Las imágenes de entrenamiento están en `dataset/`, organizadas según el formato esperado por YOLO (imágenes + anotaciones), y referenciadas desde `dataset.yaml`. Si el dataset no se incluye directamente en el repositorio por su tamaño, debe descargarse por separado y ubicarse en esta misma carpeta antes de entrenar.

## Uso

### 1. Entrenar el modelo

```bash
python main.py
```

Parte del modelo preentrenado `yolov8n.pt` y lo ajusta (fine-tuning) sobre el dataset propio. Los parámetros de entrenamiento están configurados para hardware limitado (CPU, batch pequeño, imágenes a 416px). Al finalizar, valida el modelo y guarda los resultados en `results/yolov8n_torre_perforacion/`.

### 2. Detección sobre una imagen

```bash
python interfaz.py
```

Carga el modelo entrenado (`best.pt`) y ejecuta la detección sobre una imagen de prueba definida dentro del script, dibujando manualmente las cajas y etiquetas de las detecciones.

### 3. Detección sobre un video

```bash
python interfaz2.py
```

Procesa `video1.mp4` frame por frame, muestra las detecciones en tiempo real y reporta el FPS de procesamiento en consola y sobre el propio video.

### 4. Detección en tiempo real desde webcam

```bash
python InterfazVideoYolo.py
```

Activa la cámara del equipo y muestra las detecciones en vivo. Presiona `ESC` para salir.

## Notas

- Los umbrales de confianza (`conf`) e IoU (`iou`) pueden ajustarse en cada script según se necesite mayor precisión o mayor recall.
- El modelo fue entrenado en CPU con parámetros reducidos por limitaciones de hardware; con GPU se pueden aumentar `epochs`, `batch` e `imgsz` para mejores resultados.
