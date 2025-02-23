import cv2
import numpy as np
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from picamera2 import Picamera2

# Configuración del modelo YOLO (asegúrate de tener el modelo correcto)
# MODEL_PATH = 'yolov8n.pt'
MODEL_PATH = 'yolo11n_ncnn_model'
model = YOLO(MODEL_PATH)

# Inicializar DeepSORT para el seguimiento de personas
deep_sort = DeepSort(max_age=60, n_init=3)

# Diccionarios para registrar tiempos y contar personas
time_in_store = defaultdict(lambda: None)
people_counter = set()

# Inicializar la cámara con Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration()  # Se usa configuración de preview
picam2.configure(config)
picam2.start()  # Iniciamos la cámara sin llamar a start_preview()

# Creamos una ventana OpenCV para mostrar los resultados
cv2.namedWindow("Detección y conteo de personas", cv2.WINDOW_NORMAL)

# Definimos la zona de entrada (puedes ajustar estas coordenadas)
# ENTRY_ZONE = ((100, 100), (500, 500))
ENTRY_ZONE = ((0, 0), (800, 800))

def is_inside_entry_zone(bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    (x1, y1), (x2, y2) = ENTRY_ZONE
    return x1 <= center_x <= x2 and y1 <= center_y <= y2

while True:
    # Capturamos un frame (en formato RGB)
    frame = picam2.capture_array()
    if frame is None:
        break

    # Convertimos a formato BGR para compatibilidad con OpenCV y YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Ejecutar inferencia con YOLO
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls == 0 and conf > 0.5:  # La clase 0 es "persona" en el conjunto COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

    # Actualizamos el seguimiento con DeepSORT
    tracks = deep_sort.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Verificamos si el track está en la zona de entrada
        if is_inside_entry_zone((x1, y1, x2 - x1, y2 - y1)):
            if time_in_store[track_id] is None:
                time_in_store[track_id] = time.time()  # Registro del tiempo de entrada
                people_counter.add(track_id)
            elapsed_time = int(time.time() - time_in_store[track_id])
        else:
            elapsed_time = 0

        # Dibujamos la caja y la información del track
        color = (0, 255, 0) if elapsed_time > 0 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {track_id} - {elapsed_time}s', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Dibujamos la zona de entrada y el contador total de personas
    cv2.rectangle(frame, ENTRY_ZONE[0], ENTRY_ZONE[1], (0, 255, 0), 2)
    cv2.putText(frame, 'Entrada', (ENTRY_ZONE[0][0], ENTRY_ZONE[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Personas Totales: {len(people_counter)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostramos el frame procesado
    cv2.imshow("Detección y conteo de personas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizamos la captura y cerramos la ventana
picam2.stop()
cv2.destroyAllWindows()