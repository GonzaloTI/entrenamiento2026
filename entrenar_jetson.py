from ultralytics import YOLO
import torch

"""
Script de re-entrenamiento optimizado para NVIDIA Jetson Nano (2GB).
Recomendaciones:
 - Modelo: 'yolov5nu.pt' (o 'yolov8n.pt'). Es el más rápido, liviano y con menos consumo de RAM.
 - imgsz: 416 (en lugar de 640). Reduce a la mitad el tiempo de inferencia en la Jetson manteniendo gran precisión.
"""

if __name__ == '__main__':
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Gráfica usada:", torch.cuda.get_device_name(0))

    # MODELOS RECOMENDADOS PARA JETSON NANO 2GB:
    # 1. "yolov5nu.pt" -> MÁXIMA COMPATIBILIDAD Y RENDIMIENTO (30-40+ FPS en TensorRT)
    # 2. "yolov8n.pt"  -> Buen balance (~15-20 FPS)
    # 3. "yolo11n.pt"  -> Más pesado por capas de atención (~10-15 FPS, puede requerir más RAM)
    
    modelo_elegido = "yolov5nu.pt" 
    
    print(f"\nIniciando entrenamiento con {modelo_elegido}...")
    model = YOLO(modelo_elegido)

    results = model.train(
        data="C:/Users/Usuario/Documents/python/yolo/entrenamiento/2026/yolo3/data.yaml",
        epochs=50,
        imgsz=416,       # 416x416 da una velocidad 2x en Jetson Nano comparado con 640x640
        batch=8,
        device=0,
        workers=0,
        cache=False,
        name="yolo5n_fragaria_jetson"
    )
