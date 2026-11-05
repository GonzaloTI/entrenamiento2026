# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best_fragaria2.pt")

# --- SELECCIÓN DE CÁMARA ---
# 0 -> Cámara integrada de la laptop
# 1 -> Primera cámara USB conectada
# 2 -> Segunda cámara USB (si aplica)
indice_camara = 0 

# Realizar VideoCaptura
cap = cv2.VideoCapture(indice_camara)

# Verificar si la cámara seleccionada se abrió correctamente
if not cap.isOpened():
    print(f"Error: No se pudo acceder a la cámara con el índice {indice_camara}.")
    print("Intenta cambiar el valor de 'indice_camara' a 0, 1 o 2.")
    exit()

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Evitar errores si la cámara se desconecta de repente
    if not ret:
        print("No se pudo recibir el fotograma de la cámara. Saliendo...")
        break

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.50)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa (tecla ESC)
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()