import cv2
import numpy as np
import time
import os
import sys

# ==============================================================================
# CONFIGURACIÓN GENERAL PARA JETSON NANO
# ==============================================================================
ENGINE_PATH = "best_fragaria2.engine"  # Nombre de tu motor TensorRT
IMGSZ = 640                          # Tamaño con el que se compiló el .engine (640 o 416)
CONF_THRES = 0.50                    # Umbral de confianza mínimo para detectar
IOU_THRES = 0.45                     # Umbral NMS para descartar solapamientos

# Selección del tipo de cámara:
USAR_CAMARA_CSI = False              # True = Cámara Ribbon CSI (ej. Raspberry Pi v2 / IMX219)
INDICE_CAMARA_USB = 0                # 0 = Primera cámara USB (/dev/video0)

# Nombres de las clases según data.yaml
NOMBRES_CLASES = [
    'Podredumbre de cuero',
    'Mancha negra (Antracnosis)',
    'Moho gris',
    'Oído',
    'Fruta sana'
]

# Paleta de colores vivos y contrastados para cada clase (Formato BGR de OpenCV)
COLORES = [
    (0, 0, 230),     # Podredumbre de cuero: Rojo intenso
    (0, 140, 255),   # Mancha negra: Naranja brillante
    (200, 0, 200),   # Moho gris: Magenta / Púrpura
    (255, 220, 0),   # Oído: Amarillo / Cian claro
    (0, 220, 0),     # Fruta sana: Verde brillante
]

# Pipeline de GStreamer optimizado para cámaras CSI en Jetson Nano
def obtener_pipeline_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true"
    )

# ==============================================================================
# DIBUJO DE DETECCIONES DE ALTA VISIBILIDAD
# ==============================================================================
def dibujar_deteccion(frame, x1, y1, x2, y2, clase_id, conf):
    """Dibuja rectángulos nítidos y etiquetas con fondo sólido para máxima legibilidad."""
    color = COLORES[clase_id % len(COLORES)]
    nombre = NOMBRES_CLASES[clase_id] if clase_id < len(NOMBRES_CLASES) else f"Clase {clase_id}"

    # 1. Bounding box principal
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2. Pequeños esquineros más gruesos para un aspecto visual profesional
    longitud_esquina = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
    grosor_esquina = 4
    cv2.line(frame, (x1, y1), (x1 + longitud_esquina, y1), color, grosor_esquina)
    cv2.line(frame, (x1, y1), (x1, y1 + longitud_esquina), color, grosor_esquina)
    cv2.line(frame, (x2, y1), (x2 - longitud_esquina, y1), color, grosor_esquina)
    cv2.line(frame, (x2, y1), (x2, y1 + longitud_esquina), color, grosor_esquina)
    cv2.line(frame, (x1, y2), (x1 + longitud_esquina, y2), color, grosor_esquina)
    cv2.line(frame, (x1, y2), (x1, y2 - longitud_esquina), color, grosor_esquina)
    cv2.line(frame, (x2, y2), (x2 - longitud_esquina, y2), color, grosor_esquina)
    cv2.line(frame, (x2, y2), (x2, y2 - longitud_esquina), color, grosor_esquina)

    # 3. Etiqueta con fondo sólido de color y texto blanco
    texto = f"{nombre} {int(conf * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.55
    grosor_texto = 2
    (tw, th), baseline = cv2.getTextSize(texto, font, escala, grosor_texto)

    # Posicionar etiqueta encima del cuadro (o dentro si se sale por arriba)
    y_etiqueta = y1 - 8 if y1 - th - 10 > 0 else y1 + th + 12
    cv2.rectangle(frame, (x1, y_etiqueta - th - 6), (x1 + tw + 8, y_etiqueta + 2), color, -1)
    cv2.putText(frame, texto, (x1 + 4, y_etiqueta - 4), font, escala, (255, 255, 255), grosor_texto, cv2.LINE_AA)

# ==============================================================================
# MOTOR DE INFERENCIA NATIVO TENSORRT (Para JetPack 4.6 sin Ultralytics)
# ==============================================================================
class TensorRTInference:
    def __init__(self, engine_path, imgsz=640):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.cuda = cuda
        self.imgsz = imgsz
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Cargar runtime y deserializar el motor .engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Reservar buffers de memoria Host (CPU) y Device (GPU)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocesar(self, img_bgr):
        # Redimensionar conservando aspecto con letterbox o resize directo
        h, w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalizar de [0, 255] a [0.0, 1.0] y formato NCHW
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.ascontiguousarray(np.expand_dims(img_chw, axis=0))
        return img_batch, w, h

    def inferir(self, frame, conf_thres=0.5, iou_thres=0.45):
        img_input, orig_w, orig_h = self.preprocesar(frame)
        
        # Copiar imagen a la GPU
        np.copyto(self.inputs[0]['host'], img_input.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Ejecutar modelo
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Traer resultado de GPU a CPU
        self.cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Decodificar tensor de salida de YOLO: shape (1, 4 + num_clases, num_anchors)
        raw_output = self.outputs[0]['host']
        num_classes = len(NOMBRES_CLASES)
        anchors = int(len(raw_output) / (4 + num_classes))
        output = raw_output.reshape((4 + num_classes, anchors)).T  # (anchors, 4 + num_clases)

        boxes = output[:, :4]
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filtrar por confianza
        mask = confidences >= conf_thres
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convertir de (cx, cy, w, h) a (x1, y1, w, h) escalado al tamaño original
        x_scale = orig_w / float(self.imgsz)
        y_scale = orig_h / float(self.imgsz)

        boxes_xywh = []
        for b in boxes:
            cx, cy, bw, bh = b
            x1 = int((cx - bw / 2.0) * x_scale)
            y1 = int((cy - bh / 2.0) * y_scale)
            w = int(bw * x_scale)
            h = int(bh * y_scale)
            boxes_xywh.append([x1, y1, w, h])

        # Supresión de No Máximos (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_thres, iou_thres)

        detecciones = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, bw, bh = boxes_xywh[idx]
                detecciones.append({
                    'box': (max(0, bx), max(0, by), min(orig_w, bx + bw), min(orig_h, by + bh)),
                    'conf': float(confidences[idx]),
                    'class_id': int(class_ids[idx])
                })
        return detecciones

# ==============================================================================
# BUCLE PRINCIPAL DE DETECCIÓN
# ==============================================================================
def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"\n[ERROR] No se encontró el archivo '{ENGINE_PATH}'.")
        print("Asegúrate de haber generado el motor en la Jetson con 'trtexec'.")
        return

    print("=" * 60)
    print("🍓 DETECCIÓN EN TIEMPO REAL - NVIDIA JETSON NANO (.ENGINE)")
    print("=" * 60)

    # 1. Intentar cargar con Ultralytics si está instalado, o usar motor nativo TensorRT
    usar_ultralytics = False
    try:
        from ultralytics import YOLO
        print("[INFO] Librería 'ultralytics' detectada. Cargando motor con YOLO...")
        model = YOLO(ENGINE_PATH, task="detect")
        usar_ultralytics = True
    except (ImportError, Exception):
        print("[INFO] Usando inferencia nativa TensorRT (PyCUDA + TensorRT)...")
        model = TensorRTInference(ENGINE_PATH, imgsz=IMGSZ)

    # 2. Inicializar cámara
    if USAR_CAMARA_CSI:
        print("[INFO] Conectando a cámara CSI (GStreamer)...")
        cap = cv2.VideoCapture(obtener_pipeline_csi(), cv2.CAP_GSTREAMER)
    else:
        print(f"[INFO] Conectando a cámara USB (Índice {INDICE_CAMARA_USB})...")
        cap = cv2.VideoCapture(INDICE_CAMARA_USB)

    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara. Verifica la conexión.")
        return

    # Variables para cálculo de FPS en tiempo real
    fps_anterior = 0
    fps = 0
    contador_frames = 0
    tiempo_inicio = time.time()

    print("[INFO] Sistema listo. Presiona la tecla 'ESC' o 'Q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo capturar el fotograma. Saliendo...")
            break

        # Inferencia
        if usar_ultralytics:
            resultados = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
            boxes = resultados[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    dibujar_deteccion(frame, x1, y1, x2, y2, cls_id, conf)
        else:
            detecciones = model.inferir(frame, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
            for det in detecciones:
                x1, y1, x2, y2 = det['box']
                dibujar_deteccion(frame, x1, y1, x2, y2, det['class_id'], det['conf'])

        # Cálculo de FPS
        contador_frames += 1
        tiempo_actual = time.time()
        if tiempo_actual - tiempo_inicio >= 1.0:
            fps = contador_frames / (tiempo_actual - tiempo_inicio)
            contador_frames = 0
            tiempo_inicio = tiempo_actual

        # Mostrar FPS en la esquina superior izquierda
        cv2.rectangle(frame, (10, 10), (140, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Mostrar ventana
        cv2.imshow("DETECCION JETSON NANO (TENSORRT)", frame)

        # Salir con ESC (27) o 'q'
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27 or tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado.")

if __name__ == "__main__":
    main()
