import cv2
import numpy as np
import time
import os
import sys
import ctypes

# ==============================================================================
# CONFIGURACIÓN GENERAL PARA JETSON NANO
# ==============================================================================
ENGINE_PATH = "best_fragaria2.engine"  # Nombre de tu motor TensorRT
IMGSZ = 640                          # Tamaño con el que se compiló el .engine (640 o 416)
CONF_THRES = 0.50                    # Umbral de confianza mínimo para detectar (50%)
IOU_THRES = 0.45                     # Umbral NMS para descartar cajas solapadas

# Selección de cámara:
USAR_CAMARA_CSI = False              # True = Cámara Ribbon CSI (ej. Raspberry Pi v2 / IMX219)
INDICE_CAMARA_USB = 0                # 0 = Cámara USB (/dev/video0)

# Nombres de las clases según data.yaml
NOMBRES_CLASES = [
    'Podredumbre de cuero',
    'Mancha negra (Antracnosis)',
    'Moho gris',
    'Oído',
    'Fruta sana'
]

# Paleta de colores vivos y contrastados (BGR)
COLORES = [
    (0, 0, 230),     # Podredumbre de cuero: Rojo intenso
    (0, 140, 255),   # Mancha negra: Naranja brillante
    (200, 0, 200),   # Moho gris: Magenta / Púrpura
    (255, 220, 0),   # Oído: Amarillo brillante
    (0, 220, 0),     # Fruta sana: Verde brillante
]

# Pipeline de GStreamer acelerado por hardware para cámaras CSI (IMX219)
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

def nms_numpy(boxes, scores, iou_threshold=0.45):
    """Supresión de No Máximos (NMS) en NumPy puro sin depender de cv2.dnn."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# ==============================================================================
# DIBUJO DE DETECCIONES DE ALTA VISIBILIDAD
# ==============================================================================
def dibujar_deteccion(frame, x1, y1, x2, y2, clase_id, conf):
    """Dibuja cuadros y etiquetas con fondo sólido para máxima legibilidad."""
    color = COLORES[clase_id % len(COLORES)]
    nombre = NOMBRES_CLASES[clase_id] if clase_id < len(NOMBRES_CLASES) else f"Clase {clase_id}"

    # 1. Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2. Esquinas reforzadas
    len_esq = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
    grosor_esq = 4
    cv2.line(frame, (x1, y1), (x1 + len_esq, y1), color, grosor_esq)
    cv2.line(frame, (x1, y1), (x1, y1 + len_esq), color, grosor_esq)
    cv2.line(frame, (x2, y1), (x2 - len_esq, y1), color, grosor_esq)
    cv2.line(frame, (x2, y1), (x2, y1 + len_esq), color, grosor_esq)
    cv2.line(frame, (x1, y2), (x1 + len_esq, y2), color, grosor_esq)
    cv2.line(frame, (x1, y2), (x1, y2 - len_esq), color, grosor_esq)
    cv2.line(frame, (x2, y2), (x2 - len_esq, y2), color, grosor_esq)
    cv2.line(frame, (x2, y2), (x2, y2 - len_esq), color, grosor_esq)

    # 3. Etiqueta con fondo sólido
    texto = f"{nombre} {int(conf * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.55
    grosor_texto = 2
    (tw, th), _ = cv2.getTextSize(texto, font, escala, grosor_texto)

    y_label = y1 - 8 if y1 - th - 10 > 0 else y1 + th + 12
    cv2.rectangle(frame, (x1, y_label - th - 6), (x1 + tw + 8, y_label + 2), color, -1)
    cv2.putText(frame, texto, (x1 + 4, y_label - 4), font, escala, (255, 255, 255), grosor_texto, cv2.LINE_AA)

# ==============================================================================
# MOTOR TENSORRT NATIVO (SIN NECESIDAD DE PYCUDA)
# Utiliza ctypes directamente con libcudart.so nativo de JetPack
# ==============================================================================
class TensorRTInference:
    def __init__(self, engine_path, imgsz=640):
        import tensorrt as trt

        self.imgsz = imgsz
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Cargar libcudart.so nativo de NVIDIA CUDA
        try:
            self.cudart = ctypes.CDLL("libcudart.so")
        except OSError:
            self.cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")

        # Funciones de CUDA Runtime API
        self.cudaMalloc = self.cudart.cudaMalloc
        self.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudaMalloc.restype = ctypes.c_int

        self.cudaMemcpy = self.cudart.cudaMemcpy
        self.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cudaMemcpy.restype = ctypes.c_int

        self.cudaFree = self.cudart.cudaFree
        self.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudaFree.restype = ctypes.c_int

        self.cudaMemcpyHostToDevice = 1
        self.cudaMemcpyDeviceToHost = 2

        # Cargar runtime y deserializar el motor .engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Reservar memoria GPU para inputs y outputs
        self.bindings = []
        self.input_alloc = {}
        self.output_alloc = {}

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            nbytes = int(size * np.dtype(dtype).itemsize)

            d_ptr = ctypes.c_void_p()
            status = self.cudaMalloc(ctypes.byref(d_ptr), nbytes)
            if status != 0:
                raise RuntimeError(f"Error asignando memoria CUDA (código: {status})")

            self.bindings.append(int(d_ptr.value))

            if self.engine.binding_is_input(binding):
                self.input_alloc = {
                    'd_ptr': d_ptr,
                    'nbytes': nbytes,
                    'shape': shape,
                    'dtype': dtype
                }
            else:
                self.output_alloc = {
                    'd_ptr': d_ptr,
                    'nbytes': nbytes,
                    'shape': shape,
                    'dtype': dtype
                }
                self.host_output = np.empty(shape, dtype=dtype)

    def preprocesar(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.ascontiguousarray(np.expand_dims(img_chw, axis=0))
        return img_batch, w, h

    def inferir(self, frame, conf_thres=0.5, iou_thres=0.45):
        img_batch, orig_w, orig_h = self.preprocesar(frame)

        # 1. Copiar Host -> Device (GPU)
        self.cudaMemcpy(
            self.input_alloc['d_ptr'],
            img_batch.ctypes.data_as(ctypes.c_void_p),
            self.input_alloc['nbytes'],
            self.cudaMemcpyHostToDevice
        )

        # 2. Ejecutar inferencia en la GPU
        self.context.execute_v2(self.bindings)

        # 3. Copiar Device -> Host (CPU)
        self.cudaMemcpy(
            self.host_output.ctypes.data_as(ctypes.c_void_p),
            self.output_alloc['d_ptr'],
            self.output_alloc['nbytes'],
            self.cudaMemcpyDeviceToHost
        )

        # 4. Decodificar salida YOLO: shape (1, 4 + num_clases, num_anchors)
        output = self.host_output[0].T  # (num_anchors, 4 + num_clases)

        boxes = output[:, :4]
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= conf_thres
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        x_scale = orig_w / float(self.imgsz)
        y_scale = orig_h / float(self.imgsz)

        cx = boxes[:, 0] * x_scale
        cy = boxes[:, 1] * y_scale
        bw = boxes[:, 2] * x_scale
        bh = boxes[:, 3] * y_scale

        x1 = np.clip(cx - bw / 2.0, 0, orig_w).astype(np.float32)
        y1 = np.clip(cy - bh / 2.0, 0, orig_h).astype(np.float32)
        x2 = np.clip(cx + bw / 2.0, 0, orig_w).astype(np.float32)
        y2 = np.clip(cy + bh / 2.0, 0, orig_h).astype(np.float32)

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])
        indices = nms_numpy(boxes_xyxy, confidences, iou_thres)

        detecciones = []
        for idx in indices:
            detecciones.append({
                'box': (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                'conf': float(confidences[idx]),
                'class_id': int(class_ids[idx])
            })
        return detecciones

    def __del__(self):
        # Liberar memoria GPU al salir
        if hasattr(self, 'cudaFree'):
            if self.input_alloc and 'd_ptr' in self.input_alloc:
                self.cudaFree(self.input_alloc['d_ptr'])
            if self.output_alloc and 'd_ptr' in self.output_alloc:
                self.cudaFree(self.output_alloc['d_ptr'])

# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================
def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"\n[ERROR] No se encontró el archivo '{ENGINE_PATH}'.")
        print("Asegúrate de tener el motor compilado en esta carpeta.")
        return

    print("=" * 60)
    print("🍓 DETECCIÓN EN TIEMPO REAL - NVIDIA JETSON NANO (.ENGINE)")
    print("=" * 60)

    # 1. Cargar motor TensorRT
    print("[INFO] Cargando motor TensorRT nativo en GPU...")
    try:
        model = TensorRTInference(ENGINE_PATH, imgsz=IMGSZ)
        print("[INFO] ¡Motor TensorRT cargado con éxito!")
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar TensorRT: {e}")
        return

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

    fps = 0
    contador_frames = 0
    tiempo_inicio = time.time()

    print("[INFO] Sistema listo. Presiona la tecla 'ESC' o 'Q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo capturar el fotograma. Saliendo...")
            break

        # Inferencia y dibujo
        detecciones = model.inferir(frame, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        for det in detecciones:
            x1, y1, x2, y2 = det['box']
            dibujar_deteccion(frame, x1, y1, x2, y2, det['class_id'], det['conf'])

        # Cálculo de FPS en tiempo real
        contador_frames += 1
        tiempo_actual = time.time()
        if tiempo_actual - tiempo_inicio >= 1.0:
            fps = contador_frames / (tiempo_actual - tiempo_inicio)
            contador_frames = 0
            tiempo_inicio = tiempo_actual

        # Visualizador de FPS
        cv2.rectangle(frame, (10, 10), (140, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Mostrar imagen
        cv2.imshow("DETECCION JETSON NANO (TENSORRT)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27 or tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado.")

if __name__ == "__main__":
    main()
