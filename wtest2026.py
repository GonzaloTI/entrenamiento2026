from ultralytics import YOLO
import torch

# Todo el código ejecutable debe ir dentro de este 'if'
if __name__ == '__main__':
    
    # Verificar que tu PC reconozca la gráfica
    print("CUDA disponible: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Gráfica usada:", torch.cuda.get_device_name(0))

    # Cargar el modelo base
    model = YOLO("yolo11n.pt")

    # Iniciar entrenamiento
    results = model.train(
        data="C:/Users/Usuario/Documents/python/yolo/entrenamiento/2026/yolo/data.yaml",
        epochs=40,  #anterior 15
        imgsz=640,
        batch=8,   #anterior 4
        device=0,
        workers=0,  # Evita que Windows cree subprocesos duplicados
        cache=False,     # <- Evita que YOLO llene la RAM con fotos viejas
        name="yolo_custom_train_basura2026"
    )