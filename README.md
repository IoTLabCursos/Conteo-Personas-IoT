# Conteo de Personas con IoT y Visión Artificial

## Descripción
Este proyecto utiliza **YOLO** para la detección de personas y **DeepSORT** para el seguimiento de individuos en un área específica. La implementación está diseñada para funcionar con **Raspberry Pi y Picamera2**, permitiendo el conteo de personas y el registro del tiempo que permanecen en un local.

## Características
- **Detección en tiempo real** de personas con el modelo YOLO.
- **Seguimiento de individuos** mediante el algoritmo DeepSORT.
- **Registro del tiempo** que cada persona pasa dentro del área de interés.
- **Interfaz visual en OpenCV** para mostrar las detecciones y el conteo.
- **Compatible con Raspberry Pi y Picamera2**.

## Requisitos
Asegúrate de tener los siguientes paquetes instalados en tu entorno:

```sh
pip install opencv-python numpy ultralytics deep_sort_realtime picamera2
```

## Instalación y Configuración
1. **Clonar el repositorio:**
   ```sh
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio
   ```

2. **Configurar la Picamera2:**
   - Asegúrate de que la cámara está habilitada en la Raspberry Pi.
   - Usa `raspi-config` para activarla si es necesario.

3. **Descargar o especificar el modelo YOLO:**
   - El código usa `yolo11n_ncnn_model` por defecto.

## Exportación del Modelo a NCNN
Si necesitas exportar un modelo YOLO a NCNN para su uso en dispositivos con limitaciones de hardware, sigue estos pasos:

1. **Convertir el modelo YOLO a ONNX:**
   ```sh
   yolo export model=yolov11n.pt format=ncnn
   ```
   
2. **Usar el modelo en el código:**
   - Asegúrate de cambiar `MODEL_PATH` en el script para usar el modelo optimizado.

## Uso
Ejecuta el siguiente comando para iniciar la detección y conteo:

```sh
python conteopersonas_IoT.py
```

## Explicación del Código
- **Captura de video:** Se usa `Picamera2` para obtener los fotogramas en tiempo real.
- **Detección con YOLO:** Se procesan los fotogramas y se extraen las personas detectadas.
- **Seguimiento con DeepSORT:** Cada persona detectada se asocia con un identificador único.
- **Registro del tiempo:** Se calcula cuánto tiempo cada persona permanece dentro del área definida.
- **Interfaz visual:** Se usa OpenCV para mostrar la detección y el conteo en una ventana.

## Parámetros Configurables
Puedes modificar estos valores en el código para ajustar la detección:

```python
ENTRY_ZONE = ((0, 0), (800, 800))  # Define la zona de entrada
MODEL_PATH = 'yolo11n_ncnn_model'  # Modelo YOLO a utilizar
CONFIDENCE_THRESHOLD = 0.5  # Umbral de confianza para detecciones
```

## Controles
- **Presiona 'q'** para cerrar la ventana y detener el programa.

## Licencia
Este proyecto está bajo la licencia MIT. Siéntete libre de modificarlo y mejorarlo.

## Autor
Creado por **IoTLabCursos**.

---

¡Esperamos que este proyecto sea útil para tus necesidades de conteo de personas con IoT y visión artificial! 🚀

